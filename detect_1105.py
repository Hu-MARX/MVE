import os
import os.path as osp

import cv2
import torch

import numpy as np
from AIDetector_pytorch import Detector,Detector_s

import json
import time
from mmengine import ProgressBar
from pathlib import Path
from utils.matching_pure import matching, calculate_cent_corner_pst, draw_matchingpoints, calculate_cent_corner_pst_det
from utils.common import compute_f,mve_nms1,mve_nms3,mve_nms4,mve_nms2,mve_nms1_1,mve_nms1_2
from utils.trans_matrix import supp_compute_transf_matrix as compute_transf_matrix
from utils.trans_matrix import supp_compute_transf_matrix1 as compute_transf_matrix1


# 类别映射
category = {'car': 3, 'bus': 2, 'person': 1, 'bicycle': 4}


def draw_boxes(image, boxes, confidence_threshold=0.5, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制边界框和置信度分数。

    :param image: 要绘制的图像 (numpy array)
    :param boxes: 边界框数组，形状为 (N, 6)，每行格式为 [x1, y1, x2, y2, confidence, class_id]
    :param confidence_threshold: 绘制边界框的最低置信度阈值
    :param color: 边界框颜色 (B, G, R)
    :param thickness: 边界框线条粗细
    :return: 绘制后的图像
    """
    for box in boxes:
        x1, y1, x2, y2, confidence, _ = box  # 忽略 class_id
        if confidence < confidence_threshold:
            continue  # 跳过低置信度的检测结果
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{confidence:.2f}"

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        """
        # 绘制置信度背景
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - baseline),
                      (x1 + text_width, y1), color, -1)

        # 绘制置信度文本
        cv2.putText(image, label, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        """
    return image
##boxes_image2不排序。取所有，不是前五个
# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量处理图像序列和视频进行目标检测与跟踪")
    parser.add_argument('--input1', default='/home/ubuntu/PycharmProject/Dataset/MDMT/test/1/', help='输入的父目录路径')
    parser.add_argument('--input2', default='/home/ubuntu/PycharmProject/Dataset/MDMT/test/2/', help='输入的父目录路径')
    parser.add_argument('--output', default='/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/', help='输出的父目录路径')
    parser.add_argument('--method', default='NMS-MAP-yolov8-two-detector-valid', help='the output directory name used in result_dir')
    parser.add_argument('--result_dir', default='./json_resultfiles2/supplement_supplement',help='result_dir name, no "/" in the end')
    args = parser.parse_args()

    parent_input1_dir = args.input1  # 例如: '/home/ubuntu/Datasets/MDMT/test/1/'
    parent_input2_dir = args.input2
    parent_output_dir = args.output  # 例如: 'outputs'

    # 初始化 detector 和 tracker
    # 请替换为实际的初始化代码
    # detector = YourDetectorModel()
    # tracker = StrongSORTTracker()
    # 这里使用 DummyDetector 和 DummyTracker 作为示例

    detector = Detector()
    detector2 = Detector_s()


    for dirrr in sorted(os.listdir(parent_input1_dir)):
        # 跳过包含 "-2" 的文件夹
        if "-2" in dirrr:
            print(f"Skipping {dirrr} because it contains '-2'")
            continue
        # 仅处理名称以 "-1" 结尾的文件夹
        if not dirrr.endswith("-1"):
            print(f"Skipping {dirrr} because it does not end with '-1'")
            continue

        # 构建对应的 "-2" 子文件夹的名称
        dirrr2 = dirrr.replace("-1", "-2")
        sequence_dir1 = os.path.join(parent_input1_dir, dirrr)
        sequence_dir2 = os.path.join(parent_input2_dir, dirrr2)

        # 检查 "-1" 子文件夹是否存在
        if not osp.isdir(sequence_dir1):
            print(f"警告: 文件夹 {sequence_dir1} 不存在或不是一个目录。")
            continue

        # 检查对应的 "-2" 子文件夹是否存在
        if not osp.isdir(sequence_dir2):
            print(f"警告: 对应的 '-2' 文件夹 {sequence_dir2} 不存在。")
            continue

        # 确定输出子目录
        output_subdir = os.path.join(parent_output_dir, dirrr)
        if not osp.exists(output_subdir):
            os.makedirs(output_subdir)

        if osp.isdir(sequence_dir1):
            imgs = sorted(
                filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                       # os.listdir(args.input)),
                       os.listdir(sequence_dir1)),
                key=lambda x: int(x.split('.')[0]))
        # 判断是处理图像序列还是视频
        #image_files11 = [f for f in os.listdir(sequence_dir1) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        #image_files22 = [f for f in os.listdir(sequence_dir2) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        result_dict_det = {}
        result_dict_det2 = {}
        time_start_all = time.time()
        prog_bar = ProgressBar(len(imgs))
        for i, img in enumerate(imgs):
            flag = 0
            coID_confirme = []
            supplement_bbox = np.array([])
            supplement_bbox2 = np.array([])
            if isinstance(img, str):
                # img = osp.join(args.input, img)
                img = osp.join(sequence_dir1, img)
                img2 = img.replace("/1/", "/2/")
                img2 = img2.replace("-1", "-2")
                # print(img2)
                image1 = cv2.imread(img)
                image2 = cv2.imread(img2)
            if i == 0 :
                sequence1 = img.split("/")[-2]
                sequence2 = img2.split("/")[-2]
            # 检测当前帧中的物体
            _,det_boxes1 = detector.detect(image1)
            _,det_boxes2 = detector.detect(image2)

            i = i + 1
            result_dict_det["frame={}".format(i)] = [
                [i] +  bbox.tolist() for bbox in det_boxes1
            ]

            result_dict_det2["frame={}".format(i)] = [
                [i] +  bbox.tolist() for bbox in det_boxes2
            ]
            i = i - 1

            # 绘制检测框（仅绘制置信度）
            image1_with_boxes = draw_boxes(
                image1.copy(), det_boxes1, confidence_threshold=0, color=(0, 0, 255),
                thickness=2)
            image2_with_boxes = draw_boxes(
                image2.copy(), det_boxes2, confidence_threshold=0, color=(0, 0, 255),
                thickness=2)

            # 保存绘制后的图像

            cv2.imwrite("/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_ori_A-{}.jpg".format(dirrr, i), image1_with_boxes)
            cv2.imwrite("/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_ori_B-{}.jpg".format(dirrr, i), image2_with_boxes)

            if i == 0:
                f1_det = compute_f(image1, image2)
                f2_det = compute_f(image2, image1)

                f1_last_det = f1_det
                f2_last_det = f2_det

                change_det = np.array([])
                change_det2 = np.array([])
                det_delete = np.array([])
                adjusted_boxes2 = np.array([])
                thresh = 0.2

                if det_boxes1 is not None and isinstance(det_boxes1, np.ndarray) and det_boxes1.size > 0:
                    det_boxes1, change_det, det_delete, adjusted_boxes2 = mve_nms1(img, image1, image2, det_boxes1, det_boxes2, f2_det, thresh, 100, 0.6, 0.9, detector2,0.6)
                #det_boxes2, change_det, det_delete, adjusted_boxes2 = mve_nms1(img2, image2, image1, det_boxes2,
                #                                                               det_boxes1, f1_det, thresh, 100, 0.6,
                #                                                               0.9, detector2, 0.6)
                #print("mmmmmmmmmmmmmmmmmmmmmmm", det_delete)
                                                  # 非极大值抑制
                if change_det.size > 0:
                    image1_change = draw_boxes(image1.copy(), change_det, 0, color=(0,0,255), thickness=2)
                    cv2.imwrite(
                        "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_promote_A-{}.jpg".format(
                            dirrr, i), image1_change)

                if det_delete.size > 0:
                    image1_delete = draw_boxes(image1.copy(), det_delete, 0, color=(0,0,255), thickness=2)
                    cv2.imwrite(
                        "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_delete_A-{}.jpg".format(
                            dirrr, i), image1_delete)

                if adjusted_boxes2.size > 0:
                    image1_adjusted = draw_boxes(image1.copy(), adjusted_boxes2, 0, color=(0, 0, 255), thickness=2)
                    cv2.imwrite(
                        "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_adjusted_A-{}.jpg".format(
                            dirrr, i), image1_adjusted)

                change_det = np.array([])
                change_det2 = np.array([])
                det_delete = np.array([])
                adjusted_boxes2 = np.array([])
                
                if det_boxes2 is not None and isinstance(det_boxes2, np.ndarray) and det_boxes2.size > 0:
                    det_boxes2, change_det, det_delete, adjusted_boxes2 = mve_nms1(img2, image2, image1, det_boxes2,
                                                                               det_boxes1, f1_det, thresh, 100, 0.6,
                                                                               0.9, detector2, 0.6)
                if change_det.size > 0:
                    image1_change = draw_boxes(image2.copy(), change_det, 0, color=(0, 0, 255), thickness=2)
                    cv2.imwrite(
                        "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_promote_B-{}.jpg".format(
                            dirrr, i), image1_change)

                if det_delete.size > 0:
                    image1_delete = draw_boxes(image2.copy(), det_delete, 0, color=(0, 0, 255), thickness=2)
                    cv2.imwrite(
                        "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_delete_B-{}.jpg".format(
                            dirrr, i), image1_delete)

                if adjusted_boxes2.size > 0:
                    image1_adjusted = draw_boxes(image2.copy(), adjusted_boxes2, 0, color=(0, 0, 255), thickness=2)
                    cv2.imwrite(
                        "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_adjusted_B-{}.jpg".format(
                            dirrr, i), image1_adjusted)

                i = i + 1
                result_dict_det["frame={}".format(i)] = [
                    [i] + bbox.tolist() for bbox in det_boxes1
                ]

                result_dict_det2["frame={}".format(i)] = [
                    [i] + bbox.tolist() for bbox in det_boxes2
                ]
                i = i - 1
                # 绘制检测框（仅绘制置信度）
                image1_with_boxes = draw_boxes(
                    image1.copy(), det_boxes1, confidence_threshold=0, color=(0, 0, 255),
                    thickness=2)
                image2_with_boxes = draw_boxes(
                    image2.copy(), det_boxes2, confidence_threshold=0, color=(0, 0, 255),
                    thickness=2)
                cv2.imwrite(
                    "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_final_A-{}.jpg".format(
                        dirrr, i), image1_with_boxes)
                cv2.imwrite(
                    "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_final_B-{}.jpg".format(
                        dirrr, i), image2_with_boxes)

                continue

            f2_det, f2_last_det = compute_transf_matrix1(f2_last_det, image2, image1, dirrr, i)
            f1_det, f1_last_det = compute_transf_matrix1(f1_last_det, image1, image2, dirrr, i)

            change_det = np.array([])
            change_det2 = np.array([])
            det_delete = np.array([])
            adjusted_boxes2 = np.array([])
            thresh = 0.2

            if det_boxes1 is not None and isinstance(det_boxes1, np.ndarray) and det_boxes1.size > 0:
                det_boxes1, change_det, det_delete,adjusted_boxes2 = mve_nms1(img, image1, image2, det_boxes1, det_boxes2, f2_det, thresh, 100, 0.6,
                                              0.9, detector2,0.6)
            #det_boxes2, change_det, det_delete, adjusted_boxes2 = mve_nms1(img2, image2, image1, det_boxes2,
            #                                                               det_boxes1, f1_det, thresh, 100, 0.6,
            #                                                               0.9, detector2, 0.6)

            # 非极大值抑制
            if change_det.size > 0:
                image1_change = draw_boxes(image1.copy(), change_det, 0, color=(0, 0, 255), thickness=2)
                cv2.imwrite(
                    "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_promote_A-{}.jpg".format(
                        dirrr, i), image1_change)

            if det_delete.size > 0:
                image1_delete = draw_boxes(image1.copy(), det_delete, 0, color=(0, 0, 255), thickness=2)
                cv2.imwrite(
                    "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_delete_A-{}.jpg".format(
                        dirrr, i), image1_delete)

            if adjusted_boxes2.size > 0:
                image1_adjusted = draw_boxes(image1.copy(), adjusted_boxes2, 0, color=(0, 0, 255), thickness=2)
                cv2.imwrite(
                    "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_adjusted_A-{}.jpg".format(
                        dirrr, i), image1_adjusted)

            change_det = np.array([])
            change_det2 = np.array([])
            det_delete = np.array([])
            adjusted_boxes2 = np.array([])
            
            if det_boxes2 is not None and isinstance(det_boxes2, np.ndarray) and det_boxes2.size > 0:
                det_boxes2, change_det, det_delete, adjusted_boxes2 = mve_nms1(img2, image2, image1, det_boxes2,
                                                                           det_boxes1, f1_det, thresh, 100, 0.6,
                                                                           0.9, detector2, 0.6)

            if change_det.size > 0:
                image1_change = draw_boxes(image2.copy(), change_det, 0, color=(0, 0, 255), thickness=2)
                cv2.imwrite(
                    "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_promote_B-{}.jpg".format(
                        dirrr, i), image1_change)

            if det_delete.size > 0:
                image1_delete = draw_boxes(image2.copy(), det_delete, 0, color=(0, 0, 255), thickness=2)
                cv2.imwrite(
                    "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_delete_B-{}.jpg".format(
                        dirrr, i), image1_delete)

            if adjusted_boxes2.size > 0:
                image1_adjusted = draw_boxes(image2.copy(), adjusted_boxes2, 0, color=(0, 0, 255), thickness=2)
                cv2.imwrite(
                    "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_adjusted_B-{}.jpg".format(
                        dirrr, i), image1_adjusted)

            i = i + 1

            result_dict_det["frame={}".format(i)] = [
                [i] + bbox.tolist() for bbox in det_boxes1
            ]

            result_dict_det2["frame={}".format(i)] = [
                [i] + bbox.tolist() for bbox in det_boxes2
            ]
            i = i - 1
            # 绘制检测框（仅绘制置信度）
            image1_with_boxes = draw_boxes(
                image1.copy(), det_boxes1, confidence_threshold=0, color=(0, 0, 255),
                thickness=2)
            image2_with_boxes = draw_boxes(
                image2.copy(), det_boxes2, confidence_threshold=0, color=(0, 0, 255),
                thickness=2)
            cv2.imwrite(
                "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_final_A-{}.jpg".format(dirrr,
                                                                                                                 i),
                image1_with_boxes)
            cv2.imwrite(
                "/home/ubuntu/PycharmProject/yolov82/workdirs_v_1/{}/boxes_final_B-{}.jpg".format(dirrr,
                                                                                                                 i),
                image2_with_boxes)
            prog_bar.update()
        time_end = time.time()
        method = args.method

        json_dir = "{}/{}/".format(args.result_dir, method)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        with open("{0}/{1}.json".format(json_dir, sequence1), "w") as f:
            json.dump(result_dict_det, f, indent=4)
            print("输出文件A写入完成！")
        with open("{0}/{1}.json".format(json_dir, sequence2), "w") as f2:
            json.dump(result_dict_det2, f2, indent=4)

            print("输出文件B写入完成！")
                #print("kkkkkkkkkkkkkkkkkkk", det_boxes1)

