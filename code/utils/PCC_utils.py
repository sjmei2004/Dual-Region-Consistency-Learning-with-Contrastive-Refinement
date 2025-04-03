import torch
from scipy.ndimage import distance_transform_edt as distance, gaussian_filter
import numpy as np
from skimage import segmentation as skimage_seg
@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

def context_mask(img, mask_ratio):
    img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2]
    # loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    # loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long()



def adjust_brightness(image_tensor, brightness_factor):
    """
    调整tensor类型图像的亮度，不改变tensor值的范围。

    参数:
    image_tensor (torch.Tensor): 图像的tensor。
    brightness_factor (float): 亮度调整因子。大于1将增加亮度，小于1将减少亮度。

    返回:
    torch.Tensor: 调整亮度后的图像tensor，保持原始范围。
    """
    # 计算tensor的最小值和最大值
    min_val = torch.min(image_tensor)
    max_val = torch.max(image_tensor)

    # 调整亮度
    image_tensor_shifted = image_tensor - min_val
    image_tensor_adjusted = image_tensor_shifted * brightness_factor
    image_tensor_scaled = image_tensor_adjusted + min_val

    # 确保值不超出原始范围
    image_tensor_adjusted = torch.clamp(image_tensor_scaled, min=min_val, max=max_val)

    return image_tensor_adjusted


def adjust_contrast(image_tensor, contrast_factor):
    """
    调整tensor类型图像的对比度，不改变tensor值的范围。

    参数:
    image_tensor (torch.Tensor): 图像的tensor。
    contrast_factor (float): 对比度调整因子。大于1将增加对比度，小于1将减少对比度。

    返回:
    torch.Tensor: 调整对比度后的图像tensor，保持原始范围。
    """
    # 计算tensor的平均值
    mean = torch.mean(image_tensor)

    # 计算tensor的标准差
    std_dev = torch.std(image_tensor)

    # 如果标准差为0，则直接返回原始tensor，避免除以0
    if std_dev == 0:
        return image_tensor

        # 标准化tensor
    image_tensor_normalized = (image_tensor - mean) / std_dev

    # 调整对比度
    image_tensor_contrast_adjusted = image_tensor_normalized * contrast_factor

    # 反标准化tensor，保持原始范围
    image_tensor_adjusted = image_tensor_contrast_adjusted * std_dev + mean

    # 确保值不超出原始范围
    image_tensor_adjusted = torch.clamp(image_tensor_adjusted, min=torch.min(image_tensor), max=torch.max(image_tensor))

    return image_tensor_adjusted
def blur_3d_image(image, sigma):
    """
    对三维图像进行高斯模糊。

    参数:
    image (numpy.ndarray): 输入的三维图像，形状为 (height, width, depth)。
    sigma (float): 高斯核的标准差。

    返回:
    numpy.ndarray: 模糊化后的三维图像。
    """
    tensor_cpu = image.cpu()

    # 然后，将张量转换为NumPy数组
    array = tensor_cpu.numpy()
    # 确保输入是三维的
    if len(array.shape) != 3:
        raise ValueError("Input must be a 3D image.")

        # 使用高斯滤波器对图像进行模糊化
    blurred_image = gaussian_filter(array, sigma=sigma)
    device = torch.device("cuda")
    tensor = torch.from_numpy(blurred_image)
    blurred_image1 = tensor.to(device)

    return blurred_image1
def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf


def sharpen_probabilities(probs, temperature):
    # 确保温度是正数
    if temperature <= 0:
        raise ValueError("Temperature must be a positive number.")

        # probs 的维度应该是 [num_images, num_classes, *spatial_dims]
    num_images, num_classes = probs.shape[0], probs.shape[1]

    # 遍历每张图片和每个像素位置
    for i in range(num_images):
        for j in range(num_classes):
            # 提取当前图片当前类别的概率分布
            prob_dist = probs[i, j, ...]

            # 应用温度缩放
            prob_dist_sharpened = np.power(prob_dist, 1.0 / temperature)

            # 重新归一化
            prob_dist_sharpened /= np.sum(prob_dist_sharpened)

            # 将锐化后的概率分布放回原位置
            probs[i, j, ...] = prob_dist_sharpened

    return probs

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool_)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf