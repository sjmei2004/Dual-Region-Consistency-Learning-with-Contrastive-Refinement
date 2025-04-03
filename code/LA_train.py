import os
import sys

from torchvision.utils import make_grid

from val import dist_test_all_case
from utils.PCC_utils import update_ema_variables
import nibabel as nib
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

from networks.Net import feamap_model, VNet

from utils import losses, ramps
from dataloaders.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler, RandomContrast, \
    ElasticDeformation, RandomGammaCorrection

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='data path')
parser.add_argument('--model', type=str, default='PCC', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16, help='number of labeled data')
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
patch_size = (112, 112, 80)

def entropy_map(p, C=2):
    # p N*C*W*H*D
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
    cls_model = feamap_model(num_classes=num_classes, ndf=64)
    cls_model.cuda()
    db_train = LAHeart(base_dir=train_data_path,
                       split='train80',  # train/val split
                       # num=args.labelnum,#16,
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),

                           ToTensor(),
                       ]))

    labelnum = args.labelnum  # default 16
    label_idx = list(range(0, 80))
    random.shuffle(label_idx)
    labeled_idxs = label_idx[:labelnum]
    unlabeled_idxs = label_idx[labelnum:80]
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
    # consistency_criterion1 = losses.mse_loss
    clqueuebg_hard = queue.Queue()
    clqueuela_hard = queue.Queue()
    clqueuebg_pos = queue.Queue()
    clqueuela_pos = queue.Queue()
    consist = 0.0
    consistl = 100.0
    Mask = torch.zeros(2, 2, 112, 112, 80).cuda()
    best_performance = 0.0
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
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
            loss_seg = F.cross_entropy(outputs[:labeled_bs, ...], label_batch[:labeled_bs])  # ce_loss(outputs_1[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice = losses.dice_loss(outputs_soft_1[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)


            supervised_loss = 0.5 * (loss_seg_dice + loss_seg)

            if args.has_consist == 1:
                consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output[labeled_bs:])
                EMap = entropy_map(outputs_soft_1[args.labeled_bs:], C=2)
                threshold = args.Ent_th + (0.95 - args.Ent_th) * ramps.sigmoid_rampup(iter_num, max_iterations)
                #修改选择确定性di的部分
                # print(threshold)
                mask = (EMap >= threshold).float()
                mask = torch.unsqueeze(mask, 1)

                mask = torch.cat((mask, mask), 1)
                Mask = mask
                Mask_1 = 1-mask
                consistency_loss = torch.sum(mask * consistency_dist) / (torch.sum(mask) + 1e-16)
                consistency_lowentropy_loss =torch.sum((1-mask) * consistency_dist) / (torch.sum(1-mask) + 1e-16) #consistency_criterion1(outputs_soft_1, outputs_soft_2)
                # consistency_loss =consistency_lowentropy_loss
            if epoch_num < 50:
                loss = supervised_loss + 0.06 * consistency_lowentropy_loss + 0.3 * consistency_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                update_ema_variables(model, ema_model, 0.99)

                iter_num = iter_num + 1
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
                writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
                writer.add_scalar('loss/loss_supervised', supervised_loss, iter_num)
                writer.add_scalar('loss/consistency_loss',consistency_loss,iter_num)

                logging.info(
                    'iteration %d : loss : %f, loss_seg: %f, loss_dice: %f, consistency_loss: %f' %
                    (iter_num, loss.item(), loss_seg.item(), loss_seg_dice.item(), consistency_loss.item()))

            else:
                if args.has_contrastive == 1:
                    anchor_bg = cls_model(outputs[labeled_bs:, 0, ...].unsqueeze(1))  # 2,2
                    anchor_la = cls_model(outputs[labeled_bs:, 1, ...].unsqueeze(1))

                    anchor_bg = F.softmax(anchor_bg, dim=1)  # dims(4,2)
                    anchor_la = F.softmax(anchor_la, dim=1)

                    if clqueuebg_hard.empty():
                        clqueuebg_hard.put(anchor_bg)
                        clqueuela_hard.put(anchor_la)
                        clqueuebg_pos.put(anchor_bg)
                        clqueuela_pos.put(anchor_la)
                    if epoch_num >= 50:
                        if consistency_loss > consist:
                            clqueuebg_hard.get()
                            clqueuela_hard.get()
                            clqueuebg_hard.put(anchor_bg)
                            clqueuela_hard.put(anchor_la)
                            consist = consistency_loss


                bg_hard = clqueuebg_hard.get()
                la_hard = clqueuela_hard.get()
                bg_pos = clqueuebg_pos.get()
                la_pos = clqueuela_pos.get()

                loss_contrast = losses.contrastive_loss(cos_sim, args.tau, anchor_bg, anchor_la, bg_hard.detach(), la_hard.detach(),bg_pos.detach(),la_pos.detach())
                clqueuebg_hard.put(bg_hard)
                clqueuela_hard.put(la_hard)
                clqueuebg_pos.put(anchor_bg)
                clqueuela_pos.put(anchor_la)

                if args.ramp_up_lambda != 0:
                    loss = supervised_loss + 0.06 * consistency_lowentropy_loss + 0.2 * loss_contrast+ 0.3 * consistency_loss
                else:
                    loss = supervised_loss + args.my_lambda * loss_contrast


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
                writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
                writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
                writer.add_scalar('loss/loss_supervised', supervised_loss, iter_num)

                if args.has_contrastive == 1:
                    writer.add_scalar('loss/loss_contrastive', loss_contrast, iter_num)

                logging.info(
                    'iteration %d : loss : %f, loss_seg: %f, loss_dice: %f, consistency_loss: %2f' %
                    (iter_num, loss.item(), loss_seg.item(), loss_seg_dice.item(), consistency_loss.item()))
                if args.has_contrastive == 1:
                    logging.info(
                        'iteration %d : supervised loss : %f, contrastive loss: %f' %
                        (iter_num, supervised_loss.item(), loss_contrast.item()))


                if iter_num % 2500 == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                ## save checkpoint
                if iter_num % 1000 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save({'model_state_dict': model.state_dict()}, save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
            if iter_num % 500 == 0:
                save_path = 'images/1'
                image = volume_batch[2, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # original_image = volume_batch[2, 0:1, :, :, 20:61:10].permute(1, 2, 3, 0).detach().cpu().numpy()
                # original_image = np.squeeze(original_image)  # 去除多余的维度
                # nib.save(nib.Nifti1Image(original_image.astype(np.float32), np.eye(4)),
                #          f'/original_image.nii.gz')

                image = outputs_soft_1[2, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_student',
                                 grid_image, iter_num)
                image = outputs_soft_2[2, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_teacher',
                                 grid_image, iter_num)
                # predicted_image = outputs_soft_1[2, 1:2, :, :, 20:61:10].permute(1, 2, 3, 0).detach().cpu().numpy()
                # predicted_image = np.squeeze(predicted_image)  # 去除多余的维度
                # nib.save(nib.Nifti1Image(predicted_image, np.eye(4)),
                #          f'/predicted_label.nii.gz')

                image = Mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/mask',
                                 grid_image, iter_num)

                image = Mask_1[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/mask_1',
                                 grid_image, iter_num)

                image = label_batch[2, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)
            if iter_num >= 2000 and iter_num % 50 == 0:
                model.eval()
                avg_metric = dist_test_all_case(model, iter_num, num_classes=num_classes,
                                                save_result=False,
                                                has_post=True)
                performance = avg_metric[0]
                mean_hd95 = avg_metric[2]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                        iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save({'model_state_dict': model.state_dict()}, save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    logging.info("save model to {}".format(save_mode_path))
                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
