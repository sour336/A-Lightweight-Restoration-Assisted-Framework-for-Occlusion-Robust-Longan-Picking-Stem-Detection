import numpy as np
from ultralytics import YOLO
import cv2
from stemrestore import restoration
import os

def process_frame(img_name, occ_img, img_bgr, occ_mask):
    # 载入预训练模型
    model = YOLO(r'./weight/Stempose.pt')
    model.cpu()
    # ## 可视化配置
    cv2.imwrite('./output/keypoint/' + img_name + '_re-img' + '.png', img_bgr)
    # 关键点 BGR 配色
    kpt_color_map = {
        0: {'name': '1', 'color': [255, 0, 0], 'radius': 6},
        1: {'name': '2', 'color': [255, 0, 0], 'radius': 6},
        2: {'name': '3', 'color': [255, 0, 0], 'radius': 6},
    }

    # 获取结果
    results = model(img_bgr, verbose=False) # verbose设置为False，不单独打印每一帧预测结果
    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)
    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.cpu().numpy().data.astype('uint32')

    kernel_occ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    occ_mask = np.array(occ_mask)
    occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_OPEN, kernel_occ, iterations=1)
    cv2.imwrite('./output/keypoint/' + img_name + '_re_mask' + '.png', occ_mask)

    for idx in range(num_bbox): # 遍历每个框
        # 该框所有关键点坐标和置信度
        bbox_keypoints = bboxes_keypoints[idx]
        # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]
            if occ_mask[kpt_y][kpt_x] != 0:
                kpt_color = [0,0,255]
            if kpt_x >= img_bgr.shape[1] or kpt_y >= img_bgr.shape[0]:
                continue
            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
            img_bgr = cv2.circle(occ_img, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

    cv2.imwrite(r'./output/keypoint/' + img_name + '_keypoint' + '.png', occ_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    return img_bgr

if __name__ == "__main__":
    occ_img = r'./img/008_real.png'
    name = os.path.splitext(os.path.basename(occ_img))[0]

    occ_img_path, img, mask = restoration(input=occ_img)
    mask = np.array(mask)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    occ_img = cv2.imread(occ_img_path)
    process_frame(name, occ_img, img, mask)
