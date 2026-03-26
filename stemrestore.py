import torch
import numpy as np
import os
from model.osrnet import OSRNet  # 导入基本网络
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


def load_img(file):
    """读取filepath处的图片"""
    occlusion_img = Image.open(file)
    occlusion_img = occlusion_img.resize((320,320),resample=Image.BICUBIC)
    size = occlusion_img.size
    to_tesnor = transforms.ToTensor()

    # 原规模,0.5规模,0.25规模
    img1 = occlusion_img
    img2 = img1.resize((size[0] *3 // 4, size[1] *3 // 4), resample=Image.BICUBIC)
    img3 = img2.resize((size[0] // 2, size[1] // 2), resample=Image.BICUBIC)
    # 转换为tensor,并增加第0维
    img1 = torch.unsqueeze(to_tesnor(img1), 0)
    img2 = torch.unsqueeze(to_tesnor(img2), 0)
    img3 = torch.unsqueeze(to_tesnor(img3), 0)
    # 组合成为batch字典输出
    batch = {'img1.0': img1, 'img0.75': img2, 'img0.5': img3}
    for k in batch:
        batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
    return batch


def restoration(input):

    device = torch.device('cpu')

    input_file = input

    # 加载网络结构
    net = torch.nn.DataParallel(OSRNet(xavier_init_all='xavier_init_all'))
    checkpoints = torch.load(r'./weight/OSRNet.pth',map_location=device)
    net.load_state_dict(checkpoints)
    net.to(device)

    # 图片处理
    batch = load_img(input_file)
    with torch.no_grad():
        db320_tensor, _, _ = net(batch['img1.0'], batch['img0.75'], batch['img0.5'])

        # 删去batchsize维度,限制到Imaes的Tensor(-1,1)
        db320 = torch.squeeze(db320_tensor, 0).clamp(-1, 1)
        db320 = (db320 + 1) / 2

        # 从tensor转换到image
        to_pil = transforms.ToPILImage()
        new_img = to_pil(db320)

        # 计算像素级差异
        diff = (torch.abs(db320_tensor - batch['img1.0']))
        diff_mean = diff.mean(dim=1, keepdim=True)
        diff_mean = diff_mean / torch.max(diff_mean)
        mask = (diff_mean > 0.08).float()
        mask_np = mask.squeeze().cpu().numpy() * 255
        mask_img = Image.fromarray(mask_np.astype(np.uint8))

    return input_file, new_img, mask_img