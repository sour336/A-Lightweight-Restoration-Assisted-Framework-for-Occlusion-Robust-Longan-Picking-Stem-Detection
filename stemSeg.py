import numpy as np
from ultralytics import YOLO
import cv2
from stemrestore import restoration
import os

def hex_to_bgr(hex_color):
    """
    将十六进制颜色转换为OpenCV的BGR格式
    支持格式: "#FF5733" 或 "FF5733"
    """
    # 去掉#号
    hex_color = hex_color.lstrip('#')

    # 长度检查
    if len(hex_color) != 6:
        raise ValueError("十六进制颜色格式错误，应为6位，如 #FF5733")

    # 转换为RGB元组 (R, G, B)
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    # OpenCV使用BGR顺序，需要反转 (B, G, R)
    return (rgb[2], rgb[1], rgb[0])

def process_frame(img_name, occ_img, img_bgr, occ_mask):
    # 载入预训练模型
    model = YOLO(r'./weight/Stemseg.pt')
    model.cpu()

    # 获取结果
    results = model(img_bgr, verbose=False) # verbose设置为False，不单独打印每一帧预测结果
    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)
    # 预测框的mask
    bboxes_masks = results[0].masks.cpu().numpy().data.astype('uint32')

    cv2.imwrite('./output/seg/img/' + img_name + '_re-img' + '.png', img_bgr)

    for idx in range(num_bbox):
        bbox_masks = bboxes_masks[idx]

        mask_stem = bbox_masks
        mask_occ = bbox_masks

        kernel_stem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_occ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        occ_mask = np.array(occ_mask)
        occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_OPEN, kernel_occ, iterations=1)
        cv2.imwrite('./output/seg/mask/' + img_name + '_re_mask' + '.png', occ_mask)

        mask_occ = mask_occ * occ_mask / 255
        mask_stem = mask_stem - mask_occ

        masks_stem = (mask_stem > 0.5).astype(np.uint8)
        masks_stem = masks_stem * 255
        masks_stem = cv2.morphologyEx(masks_stem, cv2.MORPH_OPEN, kernel_stem, iterations=1)
        cv2.imwrite('./output/seg/mask/' + img_name + '_mask_stem_'+'.png', masks_stem)

        mask_occ = (mask_occ > 0.5).astype(np.uint8)
        mask_occ = mask_occ * 255
        cv2.imwrite('./output/seg/mask/' + img_name + '_mask_occ_' + '.png', mask_occ)

        color_stem = hex_to_bgr('0000FF')
        color_occ = hex_to_bgr('FF0000')

        colored_mask_stem = occ_img.copy()

        colored_mask_stem[masks_stem == 255] = color_stem

        overlaid = cv2.addWeighted(occ_img, 1 - 0.5, colored_mask_stem, 0.5, 0)
        colored_mask_occ = overlaid.copy()
        colored_mask_occ[mask_occ == 255] = color_occ
        overlaid = cv2.addWeighted(overlaid, 1 - 0.5, colored_mask_occ, 0.5, 0)

        cv2.imwrite('./output/seg/img/' + img_name + '_seg' + '.png', overlaid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    occ_img = r'./img/008_real.png'
    name = os.path.splitext(os.path.basename(occ_img))[0]

    occ_img_path, img, mask = restoration(input=occ_img)
    mask = np.array(mask)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    occ_img = cv2.imread(occ_img_path)
    process_frame(name, occ_img, img, mask)