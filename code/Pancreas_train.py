import os
import sys

from Pancreas_val import dist_test_all_case
from utils.SCC_utils import update_ema_variables
from utils import test_patch
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import queue
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader

from networks.E2DNet import VNet_Encoder, MainDecoder, TriupDecoder, center_model_bra, VNet

from utils import losses, ramps
from dataloaders.pancreas import (Pancreas, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Pancreas/', help='data path')
parser.add_argument('--model', type=str, default='SCC', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=12, help='number of labeled data')
parser.add_argument('--seed', type=int, default=1337, help='random seed')#1337
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--has_triup', type=int, default=True, help='whether adopted triup decoder as auxiliary decoder')

# loss
parser.add_argument('--my_lambda', type=float, default=1, help='balance factor to control contrastive loss')
parser.add_argument('--tau', type=float, default=1, help='temperature of the contrastive loss')
# adopted this kind of ramp up weight for scc will further improve the performance
parser.add_argument('--ramp_up_lambda', type=float, default=1,
                    help='balance factor to control contrastive loss in a ramp up manner')
parser.add_argument('--rampup_param', type=float, default=40.0, help='rampup parameters')

parser.add_argument('--has_contrastive', type=int, default=1, help='whether use contrative loss')
parser.add_argument('--has_consist',type=int,default=1,help='wheter use consistency loss')
parser.add_argument('--only_supervised', type=int, default=0, help='whether use consist loss')
parser.add_argument('--Ent_th', type=float,default=0.75, help='entropy_threshold')
args = parser.parse_args()
exp_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

train_data_path = args.root_path
snapshot_path = "../semi_model/" + args.model + "/" + exp_time + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True  #
    cudnn.deterministic = False  #
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)

def entropy_map(p, C=2):
    # p N*C*W*H*D
    a=torch.sum(p*torch.log(p+1e-6), dim=1)
    b=torch.sum(p*torch.log(p+1e-6))
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    return y1
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.ramp_up_lambda * ramps.sigmoid_rampup(epoch, args.rampup_param)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = VNet(n_channels=1, n_filters=16, normalization='batchnorm', has_dropout=True).cuda()
    ema_model = VNet(n_channels=1, n_filters=16, normalization='batchnorm', has_dropout=True).cuda()
    for param in ema_model.parameters():
        param.detach_()  # ema_model set
    # classification model
    cls_model = center_model_bra(num_classes=num_classes, ndf=64)
    cls_model.cuda()
    db_train = Pancreas(base_dir=train_data_path,
                         split='train',
                         # num=None,
                         transform=transforms.Compose([
                             # RandomRotFlip(),
                             RandomCrop(patch_size),
                             ToTensor(),
                         ]))

    labelnum = args.labelnum  # default 16
    label_idx = list(range(0, 62))
    random.shuffle(label_idx)
    labeled_idxs = label_idx[:labelnum]
    unlabeled_idxs = label_idx[labelnum:62]
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    cos_sim = CosineSimilarity(dim=1, eps=1e-6)
    model.train()
    ema_model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    iterator = tqdm(range(max_epoch), ncols=70)
    consistency_criterion = losses.softmax_mse_loss
    consistency_criterion1 = losses.mse_loss
    clqueuebg = queue.Queue()
    clqueuela = queue.Queue()
    clqueuebgl = queue.Queue()
    clqueuelal = queue.Queue()
    consist = 0.0
    consistl = 100.0
    best_performance = 0.0
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            # print(i_batch)
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            #[4,1,112,112,80]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise

            outputs = model(volume_batch)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
            outputs_soft_1 = F.softmax(outputs, dim=1)
            outputs_soft_2 = F.softmax(ema_output, dim=1)

            ## calculate the supervised loss
            loss_seg_1 = F.cross_entropy(outputs[:labeled_bs, ...], label_batch[:labeled_bs])
            loss_seg_dice_1 = losses.dice_loss(outputs_soft_1[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            supervised_loss = 0.5 * (loss_seg_dice_1 + loss_seg_1)
            # print(label_batch[:1])
            if args.has_consist == 1:
                consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output[labeled_bs:])
                EMap = entropy_map(outputs_soft_1[args.labeled_bs:], C=2)
                threshold = args.Ent_th + (0.95 - args.Ent_th) * ramps.sigmoid_rampup(iter_num, max_iterations)
                #修改选择确定性di的部分
                # print(threshold)
                mask = (EMap >= threshold).float()
                mask = torch.unsqueeze(mask, 1)
                a = outputs_soft_1[args.labeled_bs:]
                mask = torch.cat((mask, mask), 1)
                consistency_loss = torch.sum(mask * consistency_dist) / (torch.sum(mask) + 1e-16)
                consistency_lowentropy_loss =torch.sum((1-mask) * consistency_dist) / (torch.sum(1-mask) + 1e-16)
                consistency_loss += consistency_lowentropy_loss
                #consistency_weight = get_current_consistency_weight(iter_num // 150)#consistency_criterion1(outputs_soft_1, outputs_soft_2)
            if epoch_num < 100:
                loss = supervised_loss + 0.3 * consistency_loss + 0.1 * consistency_lowentropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                update_ema_variables(model, ema_model, 0.99)

                iter_num = iter_num + 1
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_seg', loss_seg_1, iter_num)
                writer.add_scalar('loss/loss_dice', loss_seg_dice_1, iter_num)
                writer.add_scalar('loss/loss_supervised', supervised_loss, iter_num)
                writer.add_scalar('loss/consistency_loss',consistency_loss,iter_num)

                logging.info(
                    'iteration %d : loss : %f, loss_seg_1: %f, loss_dice_1: %f, consistency_loss: %f' %
                    (iter_num, loss.item(), loss_seg_1.item(), loss_seg_dice_1.item(),
                     consistency_loss.item()))

            else:

                if args.has_contrastive == 1:
                    create_center_1_bg = cls_model(outputs[labeled_bs:, 0, ...].unsqueeze(1))  # 2,2
                    create_center_1_la = cls_model(outputs[labeled_bs:, 1, ...].unsqueeze(1))

                    create_center_soft_1_bg = F.softmax(create_center_1_bg, dim=1)  # dims(4,2)
                    create_center_soft_1_la = F.softmax(create_center_1_la, dim=1)

                    if clqueuebg.empty():
                        clqueuebg.put(create_center_soft_1_bg)
                        clqueuela.put(create_center_soft_1_la)
                        clqueuebgl.put(create_center_soft_1_bg)
                        clqueuelal.put(create_center_soft_1_la)
                    if epoch_num >= 100:
                        if consistency_loss > consist:
                            clqueuebg.get()
                            clqueuela.get()
                            clqueuebg.put(create_center_soft_1_bg)
                            clqueuela.put(create_center_soft_1_la)
                            consist = consistency_loss



                lb_bg = clqueuebg.get()
                lb_la = clqueuela.get()
                lb_bgl = clqueuebgl.get()
                lb_lal = clqueuelal.get()

                loss_contrast = losses.scc_loss3(cos_sim, args.tau, create_center_soft_1_bg,
                                                  create_center_soft_1_la, lb_bg.detach(), lb_la.detach(),lb_bgl.detach(),lb_lal.detach())
                clqueuebg.put(lb_bg)
                clqueuela.put(lb_la)
                clqueuebgl.put(create_center_soft_1_bg)
                clqueuelal.put(create_center_soft_1_la)
                if args.ramp_up_lambda != 0:
                    loss = supervised_loss  + 0.3 * consistency_loss+ 0.1 * loss_contrast + 0.1 * consistency_lowentropy_loss  #
                else:
                    loss = supervised_loss #+ args.my_lambda * loss_contrast


                if args.only_supervised == 1:
                    print('only supervised')
                    loss = supervised_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                update_ema_variables(model, ema_model, 0.99)

                iter_num = iter_num + 1
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_seg', loss_seg_1, iter_num)
                writer.add_scalar('loss/loss_dice', loss_seg_dice_1, iter_num)
                writer.add_scalar('loss/loss_supervised', supervised_loss, iter_num)

                # if args.has_contrastive == 1:
                #     writer.add_scalar('loss/loss_contrastive', loss_contrast, iter_num)

                logging.info(
                    'iteration %d : loss : %f, loss_seg_1: %f, loss_dice_1: %f, consistency_loss: %2f' %
                    (iter_num, loss.item(), loss_seg_1.item(),  loss_seg_dice_1.item(),
                      consistency_loss.item()))
                if args.has_contrastive == 1:
                    logging.info(
                        'iteration %d : supervised loss : %f, contrastive loss: %f' %
                        (iter_num, supervised_loss.item(), loss_contrast.item()))


                if iter_num % 2500 == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                ## save checkpoint
                # if iter_num % 1000 == 0:
                #     save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                #     torch.save({'model_state_dict': model.state_dict()}, save_mode_path)
                #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= 500 and iter_num % 50 == 0:
                model.eval()
                dice = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas_CT')
                performance = dice
                # mean_hd95 = avg_metric[2]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                # writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                        iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    logging.info("save model to {}".format(save_mode_path))
                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
