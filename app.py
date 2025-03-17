import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import gradio as gr

from unimatch.unimatch import UniMatch
from dataloader.stereo import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@torch.no_grad()
def inference(image1, image2, task='stereo'):
    """Inference on an image pair for optical flow or stereo disparity prediction"""

    model = UniMatch(feature_channels=128,
                     num_scales=2,
                     upsample_factor=4,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=True,
                     task=task)

    model.eval()

    checkpoint_path = '/home/vedant/Documents/Projects_LinkedIn/Point_Cloud_Reconstruction/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth'

    checkpoint_flow = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_flow['model'], strict=True)

    padding_factor = 32
    attn_type = 'swin' if task == 'flow' else 'self_swin2d_cross_swin1d'
    attn_splits_list = [2, 8]
    corr_radius_list = [-1, 4]
    prop_radius_list = [-1, 1]
    num_reg_refine = 6 if task == 'flow' else 3

    # smaller inference size for faster speed
    max_inference_size = [384, 768] if task == 'flow' else [640, 960]

    transpose_img = False

    image1 = np.array(image1).astype(np.float32)
    image2 = np.array(image2).astype(np.float32)

    if len(image1.shape) == 2:  # gray image
        image1 = np.tile(image1[..., None], (1, 1, 3))
        image2 = np.tile(image2[..., None], (1, 1, 3))
    else:
        image1 = image1[..., :3]
        image2 = image2[..., :3]

    val_transform_list = [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    val_transform = transforms.Compose(val_transform_list)

    sample = {'left': image1, 'right': image2}
    sample = val_transform(sample)

    image1 = sample['left'].unsqueeze(0)  # [1, 3, H, W]
    image2 = sample['right'].unsqueeze(0)  # [1, 3, H, W]

    nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

    inference_size = [min(max_inference_size[0], nearest_size[0]), min(max_inference_size[1], nearest_size[1])]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                               align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                               align_corners=True)

    results_dict = model(image1, image2,
                         attn_type=attn_type,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         num_reg_refine=num_reg_refine,
                         task=task,
                         )

    flow_pr = results_dict['flow_preds'][-1]  # [1, 2, H, W] or [1, H, W]
    global disp_vis, disp_vis_col

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        pred_disp = F.interpolate(flow_pr.unsqueeze(1), size=ori_size,
                                  mode='bilinear',
                                  align_corners=True).squeeze(1)  # [1, H, W]
        pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        disp = pred_disp[0].cpu().numpy()
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis_col = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    return disp_vis, disp_vis_col

img1 = cv2.imread("/home/vedant/Documents/Projects_LinkedIn/Point_Cloud_Reconstruction/Data/im0.png")
img2 = cv2.imread("/home/vedant/Documents/Projects_LinkedIn/Point_Cloud_Reconstruction/Data/im1.png")

disp = inference(img1, img2)

# cv2.imwrite(r'D:\SSSIHMS\Constrained Illumination\20230817_170732test 3\test3_gray_nir.png', disp_vis)
cv2.imwrite("/home/vedant/Documents/Projects_LinkedIn/Point_Cloud_Reconstruction/Data/disp.png", disp_vis)
