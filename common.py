# _*_ coding: utf-8 _*_
# @Time    :2022/12/7 16:38
# @Author  :LiuZhihao
# @File    :common.py

import numpy as np
import cv2
import xml.etree.ElementTree as ET
import torch
from numpy.core.defchararray import center

from .matching_pure import matching, calculate_cent_corner_pst
import os
import tempfile
from PIL import Image
from shapely.geometry import Polygon

def calculate_polygon_iou(bbox1, bbox2):
    """
    计算两个多边形的IOU
    :param bbox1: 第一个多边形坐标，形状为 (4, 2)
    :param bbox2: 第二个多边形坐标，形状为 (4, 2)
    :return: IOU 值
    """
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union > 0 else 0
def compute_iou(box1, box2):
    """
    计算两个边界框之间的 IOU（交并比）。
    box1 和 box2 的格式为 [x_min, y_min, x_max, y_max]
    """
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]

    # 计算交集坐标
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # 计算交集面积
    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    inter_area = inter_width * inter_height

    # 计算每个框的面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算并集面积
    union_area = area1 + area2 - inter_area

    # 避免除以零
    if union_area == 0:
        return 0

    # 计算 IOU
    iou = inter_area / union_area
    return iou

def is_point_in_region(x, y, region):
    """
    判断点 (x, y) 是否位于指定的区域内。
    """
    return region['left'] <= x <= region['right'] and region['upper'] <= y <= region['lower']

def get_sub_image_id_for_point(x, y, sub_images):
    """
    根据点 (x, y) 查找所属的子区域 ID。
    """
    for sub_id, sub_img in sub_images.items():
        if is_point_in_region(x, y, sub_img):
            return sub_id
    return None

def split_image_single_folder(image_path, output_dir, vertical_splits=4, horizontal_splits=2):
    """
    将图片切割为多个不重叠的子图，并返回每个子图的位置信息（左上角坐标）和文件路径。

    参数:
        image_path (str): 输入图片的路径。
        output_dir (str): 切割后子图像的保存目录。
        vertical_splits (int): 纵向切割次数，默认为4（分成4列）。
        horizontal_splits (int): 横向切割次数，默认为2（分成2行）。

    返回:
        list of dict: 包含每个子图的文件路径和位置信息。
    """
    try:
        img = Image.open(image_path)
    except IOError:
        print(f"无法打开图片文件 {image_path}")
        return []

    width, height = img.size
    piece_width = width // vertical_splits
    piece_height = height // horizontal_splits

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sub_images_info = []
    tk = 0
    for h in range(horizontal_splits):
        for v in range(vertical_splits):
            left = v * piece_width
            upper = h * piece_height
            right = (v + 1) * piece_width if v < vertical_splits - 1 else width
            lower = (h + 1) * piece_height if h < horizontal_splits - 1 else height

            box = (left, upper, right, lower)
            piece = img.crop(box)
            piece_filename = os.path.join(output_dir, f"sub_image_{h}_{v}.jpg")
            piece.save(piece_filename)

            sub_images_info.append({
                'filename': piece_filename,
                'id': tk,
                'left': left,
                'upper': upper,
                'right': right,
                'lower': lower
            })
            tk = tk + 1
    return sub_images_info

def split_image_single_folder_overlap(image_path, output_dir, vertical_splits=4, horizontal_splits=2):
    """
    将图片切割为多个不重叠的子图，并返回每个子图的位置信息（左上角坐标）和文件路径。

    参数:
        image_path (str): 输入图片的路径。
        output_dir (str): 切割后子图像的保存目录。
        vertical_splits (int): 纵向切割次数，默认为4（分成4列）。
        horizontal_splits (int): 横向切割次数，默认为2（分成2行）。

    返回:
        list of dict: 包含每个子图的文件路径和位置信息。
    """
    try:
        img = Image.open(image_path)
    except IOError:
        print(f"无法打开图片文件 {image_path}")
        return []

    width, height = img.size
    piece_width = width // vertical_splits/2
    piece_height = height // horizontal_splits/2

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sub_images_info = []
    tk = 0
    for h in range(horizontal_splits+horizontal_splits-1):
        for v in range(vertical_splits+vertical_splits-1):
            left = v * piece_width
            upper = h * piece_height
            right = left + 2*piece_width
            lower = upper + 2 * piece_height

            box = (left, upper, right, lower)
            piece = img.crop(box)
            piece_filename = os.path.join(output_dir, f"sub_image_{h}_{v}.jpg")
            piece.save(piece_filename)

            sub_images_info.append({
                'filename': piece_filename,
                'id': tk,
                'left': left,
                'upper': upper,
                'right': right,
                'lower': lower
            })
            tk = tk + 1
    return sub_images_info

def calculate_intersection(bbox1, bbox2):
    """
    计算两个边界框的交集面积。

    参数:
    - bbox1: 第一个边界框，格式为 [x1, y1, x2, y2]
    - bbox2: 第二个边界框，格式为 [x1, y1, x2, y2]

    返回:
    - 交集区域的面积
    """
    # 获取两个边界框的坐标
    x1_min, y1_min, x1_max, y1_max = bbox1[:4]
    x2_min, y2_min, x2_max, y2_max = bbox2[:4]

    # 计算交集的左上角和右下角坐标
    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    # 计算交集的宽度和高度
    inter_width = inter_max_x - inter_min_x
    inter_height = inter_max_y - inter_min_y

    # 如果没有交集，返回0
    if inter_width <= 0 or inter_height <= 0:
        return 0

    # 计算交集面积
    intersection_area = inter_width * inter_height
    return intersection_area


def calculate_iou(box1, box2):
    """
    计算两个边界框之间的 IoU（Intersection over Union）。

    参数:
        box1 (list or array): [x_min, y_min, x_max, y_max, score]
        box2 (list or array): [x_min, y_min, x_max, y_max, score]

    返回:
        float: IoU 值，范围在 [0, 1] 之间。
    """
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]

    # 计算交集的坐标
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 计算交集面积
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 计算每个框的面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    # 计算 IoU
    if union_area == 0:
        return 0
    else:
        iou = inter_area / union_area
        return iou


def transform_boxes(detection_boxes, f):
    """
    将检测框数组的坐标通过转换矩阵 f 进行转换。

    :param detection_boxes: (N, 6) 的数组，每一行格式为 (ID, X1, Y1, X2, Y2, confidence)
    :param f: 3x3 的齐次坐标转换矩阵
    :return: 转换后的检测框数组，格式与输入相同
    """
    transformed_boxes = []

    for box in detection_boxes:
        ID, X1, Y1, X2, Y2, confidence = box

        # 创建齐次坐标 [X, Y, 1]
        point1 = np.array([X1, Y1, 1])
        point2 = np.array([X2, Y2, 1])

        # 通过矩阵 f 转换坐标
        transformed_point1 = np.dot(f, point1)
        transformed_point2 = np.dot(f, point2)

        # 将齐次坐标转为普通坐标
        X1_new, Y1_new = transformed_point1[:2] / transformed_point1[2]
        X2_new, Y2_new = transformed_point2[:2] / transformed_point2[2]

        # 保持ID和confidence不变，创建新的框
        transformed_box = [ID, X1_new, Y1_new, X2_new, Y2_new, confidence]
        transformed_boxes.append(transformed_box)

    return np.array(transformed_boxes)

def transform_boxes_det(detection_boxes, f):
    """
    将检测框数组的坐标通过转换矩阵 f 进行转换。

    :param detection_boxes: (N, 6) 的数组，每一行格式为 (ID, X1, Y1, X2, Y2, confidence)
    :param f: 3x3 的齐次坐标转换矩阵
    :return: 转换后的检测框数组，格式与输入相同
    """
    transformed_boxes = []

    for box in detection_boxes:
        X1, Y1, X2, Y2, confidence = box

        # 创建齐次坐标 [X, Y, 1]
        point1 = np.array([X1, Y1, 1])
        point2 = np.array([X2, Y2, 1])

        # 通过矩阵 f 转换坐标
        transformed_point1 = np.dot(f, point1)
        transformed_point2 = np.dot(f, point2)

        # 将齐次坐标转为普通坐标
        X1_new, Y1_new = transformed_point1[:2] / transformed_point1[2]
        X2_new, Y2_new = transformed_point2[:2] / transformed_point2[2]

        # 保持ID和confidence不变，创建新的框
        transformed_box = [X1_new, Y1_new, X2_new, Y2_new, confidence]
        transformed_boxes.append(transformed_box)

    return np.array(transformed_boxes)

def is_point_in_mask(point, mask):
    """
    检查一个点是否在指定的掩膜区域内
    :param point: (x, y) 点坐标
    :param mask: 二值化掩膜图像
    :return: bool 是否在掩膜区域内
    """
    x, y = int(point[0]), int(point[1])
    return mask[y, x] > 0  # 掩膜区域通常是非零值


def warp_image(image, homography_matrix):
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))
    return warped_image


def find_overlap_region(image1, image2, homography_matrix):


    warped_image1 = warp_image(image1, homography_matrix)

    mask1 = cv2.threshold(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    mask2 = cv2.threshold(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    warped_mask2 = warp_image(mask2, homography_matrix)

    overlap_mask = cv2.bitwise_and(warped_mask2, mask1)
    save_path = "/home/ubuntu/PycharmProject/Yolov7_StrongSORT_OSNet-main/image_1.jpg"
    #save_path1 = "/home/hyh/PycharmProjects/mdmt-code/demo/image_1.jpg"
    # 找到 overlap_mask 中非零像素的坐标
    coords = cv2.findNonZero(overlap_mask)
    if coords is not None:
        # 计算包含所有非零像素的最小矩形
        x, y, w, h = cv2.boundingRect(coords)
        # 提取最小矩形区域
        overlap_region = image1[y:y + h, x:x + w]
        if overlap_region.dtype != np.uint8:
            overlap_region = overlap_region.astype(np.uint8)


        cv2.imwrite(save_path, overlap_region)
        # 计算 overlap_region 和 image1 的面积之比
        # 计算 overlap_region 和 image1 的面积
        overlap_area = w * h
        image1_area = image1.shape[0] * image1.shape[1]

        # 调试信息，打印面积
        #print("overlap_area:", overlap_area)  # 输出 overlap_region 的面积
        #print("image1_area:", image1_area)  # 输出 image1 的面积

        # 计算面积比
        area_ratio = overlap_area / image1_area
        #print("area_ratio:", area_ratio)  # 输出面积比
        #cv2.imwrite(save_path1, overlap_region)

        return overlap_mask, overlap_region,x,y,area_ratio
    else:
        # 如果没有重叠区域，返回 None
        return None

def detect_on_overlap_region(model, overlap_region, x_offset, y_offset, frame_id, bboxes1, ids1, labels1, max_id):
    # 在 overlap_region 上进行目标检测
    result, max_id = inference_mot(
        model,
        overlap_region,
        frame_id=frame_id,
        bboxes1=bboxes1,
        ids1=ids1,
        labels1=labels1,
        max_id=max_id
    )

    # 检查 result['det_bboxes'] 的结构
    #print("result['det_bboxes']:", result['det_bboxes'])

    adjusted_bboxes = []

    # 获取检测框数组
    det_bboxes_array = result['det_bboxes'][0]  # 取出数组

    # 迭代每个检测框
    for bbox in det_bboxes_array:
        x_min, y_min, x_max, y_max, score = bbox  # 解包 5 个值

        # 将边界框坐标加上偏移量 (x_offset, y_offset)
        adjusted_bbox = [
            x_min + x_offset,
            y_min + y_offset,
            x_max + x_offset,
            y_max + y_offset,
            score  # 保留置信度分数
        ]
        adjusted_bboxes.append(adjusted_bbox)
    adjusted_bboxes = np.array(adjusted_bboxes)
    return adjusted_bboxes




def compute_center_of_boxes(boxes):
    centers = [(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2) for _, x1, y1, x2, y2, _ in boxes]
    return centers


def is_point_in_mask(point, mask):
    x, y = int(point[0]), int(point[1])
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        return mask[y, x] > 0
    return False

def filter_boxes_in_overlap_region(boxes_image1, centers_image1, boxes_image2, centers_image2, homography_matrix,
                                   overlap_mask):
    filtered_boxes_image1 = []
    filtered_boxes_image2 = []

    for box, center in zip(boxes_image1, centers_image1):
        center_point = np.array([[center]], dtype='float32')
        transformed_center = cv2.perspectiveTransform(center_point, homography_matrix)[0][0]
        if is_point_in_mask(transformed_center, overlap_mask):
            filtered_boxes_image1.append(box)

    for box, center in zip(boxes_image2, centers_image2):
        if is_point_in_mask(center, overlap_mask):
            filtered_boxes_image2.append(box)

    return np.array(filtered_boxes_image1), np.array(filtered_boxes_image2)

def filter_boxes_in_overlap_region_det(boxes_image1, boxes_image2, homography_matrix, overlap_mask):
    """
        筛选在重叠区域内的检测框
        :param boxes_image1: 图像1中的检测框列表，格式为 [id, x1, y1, x2, y2]
        :param boxes_image2: 图像2中的检测框列表，格式为 [id, x1, y1, x2, y2]
        :param homography_matrix: 从图像1到图像2的单应性矩阵
        :param overlap_mask: 重叠区域的掩膜
        :return: 在重叠区域内的图像1和图像2的检测框
        """
    filtered_boxes_image1 = []
    filtered_boxes_image2 = []

    # 处理图像1的检测框
    for box in boxes_image2:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2]
        ], dtype='float32')

        # 使用单应性矩阵将图像1中的角点变换到图像2的坐标系中
        transformed_corners = cv2.perspectiveTransform(np.array([corners]), homography_matrix)[0]
        if any(is_point_in_mask(corner, overlap_mask) for corner in transformed_corners):
            filtered_boxes_image2.append(box)

    # 处理图像2的检测框
    for box in boxes_image1:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2]
        ], dtype='float32')

        if any(is_point_in_mask(corner, overlap_mask) for corner in corners):
            filtered_boxes_image1.append(box)

    return np.array(filtered_boxes_image1), np.array(filtered_boxes_image2)

def filter_boxes_in_overlap_region_det_cen(boxes_image1, boxes_image2, homography_matrix, overlap_mask):
    """
    筛选在重叠区域内的检测框（根据中心点）
    :param boxes_image1: 图像1中的检测框列表，格式为 [id, x1, y1, x2, y2]
    :param boxes_image2: 图像2中的检测框列表，格式为 [id, x1, y1, x2, y2]
    :param homography_matrix: 从图像1到图像2的单应性矩阵
    :param overlap_mask: 重叠区域的掩膜
    :return: 在重叠区域内的图像1和图像2的检测框
    """
    filtered_boxes_image1 = []
    filtered_boxes_image2 = []

    # 处理图像2的检测框
    for box in boxes_image2:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_point = np.array([center_x, center_y], dtype='float32')

        # 使用单应性矩阵将图像2中的中心点变换到图像1的坐标系中
        # 修改为 (1, 2) 形状
        transformed_center = cv2.perspectiveTransform(np.array([[center_point]]), homography_matrix)[0][0]

        # 判断中心点是否在重叠区域内
        if is_point_in_mask(transformed_center, overlap_mask):
            filtered_boxes_image2.append(box)

    # 处理图像1的检测框
    for box in boxes_image1:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_point = np.array([center_x, center_y], dtype='float32')

        # 判断中心点是否在重叠区域内
        if is_point_in_mask(center_point, overlap_mask):
            filtered_boxes_image1.append(box)

    return np.array(filtered_boxes_image1), np.array(filtered_boxes_image2)

def filter_boxes_in_overlap_region1(boxes_image1, boxes_image2, homography_matrix, overlap_mask):
    """
    筛选在重叠区域内的检测框
    :param boxes_image1: 图像1中的检测框列表，格式为 [id, x1, y1, x2, y2]
    :param boxes_image2: 图像2中的检测框列表，格式为 [id, x1, y1, x2, y2]
    :param homography_matrix: 从图像1到图像2的单应性矩阵
    :param overlap_mask: 重叠区域的掩膜
    :return: 在重叠区域内的图像1和图像2的检测框
    """
    filtered_boxes_image1 = []
    filtered_boxes_image2 = []

    # 处理图像1的检测框
    for box in boxes_image1:
        x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2]
        ], dtype='float32')

        # 使用单应性矩阵将图像1中的角点变换到图像2的坐标系中
        transformed_corners = cv2.perspectiveTransform(np.array([corners]), homography_matrix)[0]
        if any(is_point_in_mask(corner, overlap_mask) for corner in transformed_corners):
            filtered_boxes_image1.append(box)

    # 处理图像2的检测框
    for box in boxes_image2:
        x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2]
        ], dtype='float32')

        if any(is_point_in_mask(corner, overlap_mask) for corner in corners):
            filtered_boxes_image2.append(box)

    return np.array(filtered_boxes_image1), np.array(filtered_boxes_image2)

def find_big_detection_boxes(detections, km=1.5):
    # 提取坐标和计算面积
    areas = (detections[:, 4] - detections[:, 2]) * (detections[:, 3] - detections[:, 1])
    #print("nnnnnnnnnnnnnnnnn", detections)
    #print("mmmmmmmmmmmmmmmmmmm", areas)
    # 找到最小面积
    min_area = np.mean(areas)

    # 计算面积阈值（1.5倍最小面积）
    area_threshold = km * min_area

    # 找到符合条件的检测框
    valid_indices = np.where(areas >= area_threshold)[0]
    valid_detections = detections[valid_indices]

    return valid_detections

def find_big_and_small_detection_boxes(detections, km=1.5):
    # 提取坐标并计算面积
    areas = (detections[:, 4] - detections[:, 2]) * (detections[:, 3] - detections[:, 1])

    # 计算平均面积作为基准
    mean_area = np.mean(areas)

    # 计算面积阈值（km倍的平均面积）
    area_threshold = km * mean_area

    # 找到符合条件的检测框（面积大于等于阈值的）
    big_indices = np.where(areas >= area_threshold)[0]
    big_detections = detections[big_indices]

    # 找到不符合条件的检测框（面积小于阈值的）
    small_indices = np.where(areas < area_threshold)[0]
    small_detections = detections[small_indices]

    return big_detections, small_detections
def find_big_and_small_detection_boxes_middle(detections):
    # 提取坐标并计算面积
    areas = (detections[:, 4] - detections[:, 2]) * (detections[:, 3] - detections[:, 1])

    # 计算最大和最小面积
    max_area = np.max(areas)
    min_area = np.min(areas)

    # 计算新的面积阈值（最大面积 + 最小面积）/ 2
    #area_threshold = (max_area + min_area) / 2
    area_threshold = np.median(areas) #中值

    # 找到符合条件的检测框（面积大于等于阈值的）
    big_indices = np.where(areas >= area_threshold)[0]
    big_detections = detections[big_indices]

    # 找到不符合条件的检测框（面积小于阈值的）
    small_indices = np.where(areas < area_threshold)[0]
    small_detections = detections[small_indices]

    return big_detections, small_detections, max_area, min_area, area_threshold
def find_big_and_small_detection_boxes_middle_det(detections):
    # 提取坐标并计算面积
    areas = (detections[:, 3] - detections[:, 1]) * (detections[:, 2] - detections[:, 0])

    # 计算最大和最小面积
    max_area = np.max(areas)
    min_area = np.min(areas)

    # 计算新的面积阈值（最大面积 + 最小面积）/ 2
    #area_threshold = (max_area + min_area) / 2
    area_threshold = np.median(areas) #中值

    # 找到符合条件的检测框（面积大于等于阈值的）
    big_indices = np.where(areas >= area_threshold)[0]
    big_detections = detections[big_indices]

    # 找到不符合条件的检测框（面积小于阈值的）
    small_indices = np.where(areas < area_threshold)[0]
    small_detections = detections[small_indices]

    return area_threshold

def compute_max_min_area(boxes):
    if boxes.shape[0] == 0:
        return 0, 0

    # boxes assumed to be in the format [id, x1, y1, x2, y2, confidence]
    # Extract coordinates
    x1 = boxes[:, 1]
    y1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]

    # Compute area of each box
    areas = (x2 - x1) * (y2 - y1)

    # Find the maximum and minimum areas
    max_area = np.max(areas)
    min_area = np.min(areas)

    return max_area, min_area




def find_overlap_and_filter_boxes(image1, image2, boxes_image1, boxes_image2, homography_matrix, k1=0.7, k2=0.7):
    confidence_threshold = 0.2
    boxes_image1_ori = boxes_image1
    overlap_mask = find_overlap_region(image1, image2, homography_matrix)
    centers_image1 = compute_center_of_boxes(boxes_image1)
    centers_image2 = compute_center_of_boxes(boxes_image2)

    #boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    #centers_image1 = compute_center_of_boxes(boxes_image1_1)
    #centers_image1 = compute_center_of_boxes(boxes_image1)

    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
                                                                                  boxes_image2, centers_image2,
                                                                                  homography_matrix, overlap_mask)
    #filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, boxes_image2,homography_matrix, overlap_mask)

    bboxes_image_f = np.array([])
    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        return boxes_image1, bboxes_image_f

    #filtered_boxes_image_big, filtered_boxes_image_small = find_big_and_small_detection_boxes(filtered_boxes_image1, 1.5)
    filtered_boxes_image_big, filtered_boxes_image_small,max_area, min_area, M = find_big_and_small_detection_boxes_middle(filtered_boxes_image1)
                                                                                              
    ids1 = filtered_boxes_image_big[:, 0]
    ids3 = filtered_boxes_image_small[:, 0]
    ids2 = filtered_boxes_image2[:, 0]

    bbox1 = filtered_boxes_image_big[~np.isin(ids1, ids2)]
    bbox2 = filtered_boxes_image_small[~np.isin(ids3, ids2)]
    bbox3 = filtered_boxes_image_small[np.isin(ids3, ids2)]
    '''
    if bbox3.shape[0] == 0:
        print("no matched small boxes found")
    else:
        # === 添加去除重复 ID 的低置信度框逻辑 ===
        # 提取 bbox3 中的 ID 和置信度列
        ids_bbox3 = bbox3[:, 0]  # bbox3 的 ID 列
        confidences_bbox3 = bbox3[:, 5]  # bbox3 的置信度列（第六列）

        # 创建一个字典，存储每个 ID 对应的最高置信度框的索引
        id_confidence_dict = {}

        for idx, (box_id, confidence) in enumerate(zip(ids_bbox3, confidences_bbox3)):
            if box_id not in id_confidence_dict:
                id_confidence_dict[box_id] = idx
            else:
                existing_idx = id_confidence_dict[box_id]
                if confidence > confidences_bbox3[existing_idx]:
                    # 如果当前框置信度更高，更新索引
                    id_confidence_dict[box_id] = idx

        # 获取需要保留的索引
        indices_to_keep = list(id_confidence_dict.values())

        # 根据索引过滤 bbox3，只保留每个 ID 最高置信度的框
        bbox3 = bbox3[indices_to_keep]

        ids4 = bbox3[:, 0]
        bbox4 = filtered_boxes_image2[np.isin(ids2, ids4)]

        ids_bbox3 = bbox3[:, 0]  # bbox3 的 ID 列
        confidence3 = bbox3[:, -1]  # bbox3 的置信度列

        ids_bbox4 = bbox4[:, 0]  # bbox4 的 ID 列
        confidence4 = bbox4[:, -1]  # bbox4 的置信度列

        # 通过 ID 进行匹配，确保 bbox3 和 bbox4 的框数量一致
        common_ids = np.intersect1d(ids_bbox3, ids_bbox4)

        # 过滤出这些共有 ID 的框
        matching_indices_bbox3 = np.where(np.isin(ids_bbox3, common_ids))[0]
        matching_indices_bbox4 = np.where(np.isin(ids_bbox4, common_ids))[0]

        # 提取匹配后的置信度
        confidence3 = confidence3[matching_indices_bbox3]
        confidence4 = confidence4[matching_indices_bbox4]

        # 找到 bbox3 中比 bbox4 置信度低的框的索引
        low_confidence_indices = np.where(confidence3 < confidence4)[0]

        # 将这些框的置信度更新为 bbox4 中较大的置信度
        bbox3[low_confidence_indices, -1] = confidence4[low_confidence_indices]

        # 提取 bbox3 的 ID 和置信度列
        ids_bbox3 = bbox3[:, 0]  # bbox3 的 ID 列
        confidence_bbox3 = bbox3[:, -1]  # bbox3 的置信度列 (保持一致使用最后一列)

        # 找到 boxes_image1 中与 bbox3 ID 相同的框
        indices_image1 = np.where(np.isin(boxes_image1[:, 0], ids_bbox3))[0]

        # 找到 bbox3 中与 boxes_image1 相匹配的 ID 的框
        # 使用 np.where 来代替 np.searchsorted 更精确地找到对应的 ID
        matching_indices_bbox3 = np.where(np.isin(ids_bbox3, boxes_image1[indices_image1, 0]))[0]

        # 更新 boxes_image1 中与 bbox3 ID 相同的框的置信度为 bbox3 的置信度 (第六列)
        boxes_image1[indices_image1, -1] = confidence_bbox3[matching_indices_bbox3]
    '''
    # 当没有大框或小框匹配时分别处理
    if bbox1.shape[0] == 0 and bbox2.shape[0] == 0:
        print("no matched boxes found")
        return boxes_image1, bboxes_image_f
    elif bbox1.shape[0] == 0:
        print("No large boxes matched")
        id_bbox2 = bbox2[:, 0]
        id_boxes_image1 = boxes_image1[:, 0]
        matching_indices2 = np.isin(id_boxes_image1, id_bbox2)
        boxes_image1[matching_indices2, 5] *= k2
        bboxes_image_f = boxes_image1[matching_indices2]
    elif bbox2.shape[0] == 0:
        print("No small boxes matched")
        id_bbox1 = bbox1[:, 0]
        id_boxes_image1 = boxes_image1[:, 0]
        matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
        boxes_image1[matching_indices1, 5] *= k1
        bboxes_image_f = boxes_image1[matching_indices1]
    else:
        # 如果同时匹配到大框和小框，分别处理
        id_bbox1 = bbox1[:, 0]
        id_bbox2 = bbox2[:, 0]

        id_boxes_image1 = boxes_image1[:, 0]

        matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
        matching_indices2 = np.isin(id_boxes_image1, id_bbox2)

        # 只缩放置信度
        boxes_image1[matching_indices1, 5] *= k1
        boxes_image1[matching_indices2, 5] *= k2

        # 合并大框和小框的检测框
        bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        # 5. 添加对置信度的筛选，置信度大于 0.2
        high_confidence_indices = boxes_image1[:, 5] > confidence_threshold
        boxes_image1 = boxes_image1[high_confidence_indices]

        if boxes_image1.shape[0] == 0:
            boxes_image1 = boxes_image1_ori

        id_11 = boxes_image1[:, 0]
        id_22 = boxes_image1_ori[:, 0]

        # 找到 boxes_image1 中与 boxes_image1_ori 中 ID 相同的框的索引
        matching_indices = np.isin(id_11, id_22)

        # 获取 boxes_image1 中匹配的框的 ID
        matching_ids = id_11[matching_indices]

        # 查找 boxes_image1_ori 中对应匹配的框
        matching_boxes_ori = np.array(
            [boxes_image1_ori[boxes_image1_ori[:, 0] == match_id] for match_id in matching_ids]).squeeze()

        # 将 boxes_image1 中匹配的框替换为 boxes_image1_ori 中对应的框
        boxes_image1[matching_indices] = matching_boxes_ori

    return boxes_image1, bboxes_image_f

'''
def find_overlap_and_filter_boxes(image1, image2, boxes_image1, boxes_image2, homography_matrix, k1=0.7, k2=0.7):
    confidence_threshold = 0.2
    boxes_image1_ori = boxes_image1
    overlap_mask = find_overlap_region(image1, image2, homography_matrix)

    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, boxes_image2,
                                                                                  homography_matrix, overlap_mask)

    bboxes_image_f = np.array([])
    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        return boxes_image1, bboxes_image_f

    # 首先计算最大面积和最小面积
    #max_area, min_area = compute_max_min_area(filtered_boxes_image1)

    # 判断最大面积是否大于最小面积的两倍
    if filtered_boxes_image1.shape[0] > 4:
        # 如果满足条件，执行find_big_and_small_detection_boxes_middle
        filtered_boxes_image_big, filtered_boxes_image_small, max_area, min_area, M = find_big_and_small_detection_boxes_middle(
            filtered_boxes_image1)

        ids1 = filtered_boxes_image_big[:, 0]
        ids3 = filtered_boxes_image_small[:, 0]
        ids2 = filtered_boxes_image2[:, 0]

        bbox1 = filtered_boxes_image_big[~np.isin(ids1, ids2)]
        bbox2 = filtered_boxes_image_small[~np.isin(ids3, ids2)]

        # 根据匹配结果分别处理大框和小框
        if bbox1.shape[0] == 0 and bbox2.shape[0] == 0:
            print("no matched boxes found")
            return boxes_image1, bboxes_image_f
        elif bbox1.shape[0] == 0:
            print("No large boxes matched")
            id_bbox2 = bbox2[:, 0]
            id_boxes_image1 = boxes_image1[:, 0]
            matching_indices2 = np.isin(id_boxes_image1, id_bbox2)
            boxes_image1[matching_indices2, 5] *= k2
            bboxes_image_f = boxes_image1[matching_indices2]
        elif bbox2.shape[0] == 0:
            print("No small boxes matched")
            id_bbox1 = bbox1[:, 0]
            id_boxes_image1 = boxes_image1[:, 0]
            matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
            boxes_image1[matching_indices1, 5] *= k1
            bboxes_image_f = boxes_image1[matching_indices1]
        else:
            # 同时处理大框和小框
            id_bbox1 = bbox1[:, 0]
            id_bbox2 = bbox2[:, 0]
            id_boxes_image1 = boxes_image1[:, 0]

            matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
            matching_indices2 = np.isin(id_boxes_image1, id_bbox2)

            boxes_image1[matching_indices1, 5] *= k1
            boxes_image1[matching_indices2, 5] *= k2

            bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)
    else:
        print("less than 5 detections")
        ids2 = filtered_boxes_image2[:, 0]
        id_boxes_image1 = boxes_image1[:, 0]  # 注意这里使用 boxes_image1 而不是 filtered_boxes_image1
        non_matching_indices = ~np.isin(id_boxes_image1, ids2)  # 基于 boxes_image1 的 ID 列生成布尔索引
        boxes_image1[non_matching_indices, 5] *= 0.8
        bboxes_image_f = boxes_image1[non_matching_indices]
    # 添加对置信度的筛选，置信度大于 0.2
    high_confidence_indices = boxes_image1[:, 5] > confidence_threshold
    boxes_image1 = boxes_image1[high_confidence_indices]

    if boxes_image1.shape[0] == 0:
        boxes_image1 = boxes_image1_ori
    return boxes_image1, bboxes_image_f

'''
'''
def find_overlap_and_filter_boxes(image1, image2, boxes_image1, boxes_image2, homography_matrix, k1=0.7, k2=0.7):
    """
    在两张图像的重叠区域内过滤检测框，并根据检测框的大小动态调整置信度。

    参数：
    - image1, image2: 输入的两张图像
    - boxes_image1, boxes_image2: 分别来自两张图像的检测框，形状为 (N, 6)
    - homography_matrix: 图像1到图像2的单应矩阵
    - k1, k2: 初始的置信度调整系数

    返回：
    - boxes_image1 (np.ndarray): 调整后的图像1的检测框
    - bboxes_image_f (np.ndarray): 过滤后的检测框
    """
    confidence_threshold = 0.2
    boxes_image1_ori = boxes_image1
    # 1. 寻找图像重叠区域
    overlap_mask = find_overlap_region(image1, image2, homography_matrix)

    # 2. 过滤位于重叠区域中的检测框
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(
        boxes_image1, boxes_image2, homography_matrix, overlap_mask
    )

    # 3. 如果没有检测框在重叠区域，直接返回原始检测框
    bboxes_image_f = np.array([])
    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        return boxes_image1, bboxes_image_f

    # 4. 分类大框和小框，并获取面积信息
    big_detections, small_detections, max_area, min_area, M = find_big_and_small_detection_boxes_middle(filtered_boxes_image1)

    # 如果没有大框或小框，继续处理
    ids1 = big_detections[:, 0]
    ids3 = small_detections[:, 0]
    ids2 = filtered_boxes_image2[:, 0]

    bbox1 = big_detections[~np.isin(ids1, ids2)]
    bbox2 = small_detections[~np.isin(ids3, ids2)]

    # 当没有大框或小框匹配时分别处理
    if bbox1.shape[0] == 0 and bbox2.shape[0] == 0:
        print("No matched boxes found")
        return boxes_image1, bboxes_image_f
    elif bbox1.shape[0] == 0:
        print("No large boxes matched")
        id_bbox2 = bbox2[:, 0]
        id_boxes_image1 = boxes_image1[:, 0]
        matching_indices2 = np.isin(id_boxes_image1, id_bbox2)
        

        # 获取对应检测框的面积
        matched_boxes_small = boxes_image1[matching_indices2]
        areas_small = (matched_boxes_small[:, 4] - matched_boxes_small[:, 2]) * (matched_boxes_small[:, 3] - matched_boxes_small[:, 1])

        # 计算动态调整的 k2
        dynamic_k2 = k2 * (M - areas_small) / M

        # 确保动态调整后的 k2 在合理范围内
        dynamic_k2 = np.clip(dynamic_k2, 0, 1)

        dynamic_k2 = np.clip(dynamic_k2, k2 / 2, 1)  # 限制 k2 的范围，并确保不小于 k2 / 2
        
        # 应用动态调整的 k2
        boxes_image1[matching_indices2, 5] *= k2
        bboxes_image_f = boxes_image1[matching_indices2]
    elif bbox2.shape[0] == 0:
        print("No small boxes matched")
        id_bbox1 = bbox1[:, 0]
        id_boxes_image1 = boxes_image1[:, 0]
        matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
        
        # 获取对应检测框的面积
        matched_boxes_big = boxes_image1[matching_indices1]
        areas_big = (matched_boxes_big[:, 4] - matched_boxes_big[:, 2]) * (matched_boxes_big[:, 3] - matched_boxes_big[:, 1])

        # 计算动态调整的 k1
        dynamic_k1 = k1 * (max_area - areas_big) / M

        # 确保动态调整后的 k1 在合理范围内
        dynamic_k1 = np.clip(dynamic_k1, 0, 1)

        dynamic_k1 = np.clip(dynamic_k1, k1 / 2, 1)  # 限制 k1 的范围，并确保不小于 k1 / 2
        
        # 应用动态调整的 k1
        boxes_image1[matching_indices1, 5] *= k1
        bboxes_image_f = boxes_image1[matching_indices1]
    else:
        # 如果同时匹配到大框和小框，分别处理
        id_bbox1 = bbox1[:, 0]
        id_bbox2 = bbox2[:, 0]

        id_boxes_image1 = boxes_image1[:, 0]

        matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
        matching_indices2 = np.isin(id_boxes_image1, id_bbox2)
        
        # 获取对应检测框的面积
        matched_boxes_big = boxes_image1[matching_indices1]
        areas_big = (matched_boxes_big[:, 4] - matched_boxes_big[:, 2]) * (matched_boxes_big[:, 3] - matched_boxes_big[:, 1])

        matched_boxes_small = boxes_image1[matching_indices2]
        areas_small = (matched_boxes_small[:, 4] - matched_boxes_small[:, 2]) * (matched_boxes_small[:, 3] - matched_boxes_small[:, 1])
        
        # 计算动态调整的 k1 和 k2
        dynamic_k1 = k1 * (max_area - areas_big) / M
        dynamic_k2 = k2 * (M - areas_small) / M

        # 确保动态调整后的 k1 和 k2 在合理范围内
        dynamic_k1 = np.clip(dynamic_k1, 0, 1)
        dynamic_k2 = np.clip(dynamic_k2, 0, 1)

        dynamic_k1 = np.clip(dynamic_k1, 2*k1 / 3, 1)  # 限制 k1 的范围，并确保不小于 k1 / 2
        dynamic_k2 = np.clip(dynamic_k2, 2*k2 / 3, 1)  # 限制 k2 的范围，并确保不小于 k2 / 2

        # 应用动态调整的 k1 和 k2
        boxes_image1[matching_indices1, 5] *= dynamic_k1
        boxes_image1[matching_indices2, 5] *= dynamic_k2
        
        boxes_image1[matching_indices1, 5] *= k1
        boxes_image1[matching_indices2, 5] *= k2

        # 合并大框和小框的检测框
        bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

    # 5. 添加对置信度的筛选，置信度大于 0.2
    high_confidence_indices = boxes_image1[:, 5] > confidence_threshold
    boxes_image1 = boxes_image1[high_confidence_indices]

    if boxes_image1.shape[0]==0:
        boxes_image1 = boxes_image1_ori

    id_11 = boxes_image1[:,0]
    id_22 = boxes_image1_ori[:,0]

    # 找到 boxes_image1 中与 boxes_image1_ori 中 ID 相同的框的索引
    matching_indices = np.isin(id_11, id_22)

    # 获取 boxes_image1 中匹配的框的 ID
    matching_ids = id_11[matching_indices]

    # 查找 boxes_image1_ori 中对应匹配的框
    matching_boxes_ori = np.array(
        [boxes_image1_ori[boxes_image1_ori[:, 0] == match_id] for match_id in matching_ids]).squeeze()

    # 将 boxes_image1 中匹配的框替换为 boxes_image1_ori 中对应的框
    boxes_image1[matching_indices] = matching_boxes_ori

    return boxes_image1, bboxes_image_f
'''

def find_unique_boxes(arr1, arr2):

    # 获取id列
    ids1 = arr1[:, 0]
    ids2 = arr2[:, 0]

    # 找到第一个数组中存在但第二个数组中不存在的id
    unique_ids = np.setdiff1d(ids1, ids2)

    # 根据id筛选出第一个数组中独有的行
    unique_boxes = arr1[np.isin(ids1, unique_ids)]

    return unique_boxes

def compute_iou(box1, box2):
    """
    计算两个边界框的IoU（交并比）
    box1, box2: [ID, xmin, ymin, xmax, ymax, confidence]
    只使用 [xmin, ymin, xmax, ymax] 来计算IoU
    """
    x1 = max(box1[1], box2[1])  # xmin
    y1 = max(box1[2], box2[2])  # ymin
    x2 = min(box1[3], box2[3])  # xmax
    y2 = min(box1[4], box2[4])  # ymax

    # 计算交集区域的宽度和高度
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # 计算每个边界框的面积
    box1_area = (box1[3] - box1[1]) * (box1[4] - box1[2])
    box2_area = (box2[3] - box2[1]) * (box2[4] - box2[2])

    # 计算并集区域的面积
    union_area = box1_area + box2_area - inter_area

    # 返回IoU
    return inter_area / union_area if union_area != 0 else 0

'''
def all_nms_plus_1(image1, image2, dets, dets2, homography_matrix, iou_thresh):
    boxes_image1 = dets
    boxes_image2 = dets2

    overlap_mask = find_overlap_region(image1, image2, homography_matrix)
    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, boxes_image2,
                                                                                  homography_matrix, overlap_mask)

    bboxes_image_f = np.array([])
    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 1]  # xmin
        y1 = boxes_image1[:, 2]  # ymin
        x2 = boxes_image1[:, 3]  # xmax
        y2 = boxes_image1[:, 4]  # ymax
        scores = boxes_image1[:, 5]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            # 选择大于x1,y1和小于x2,y2的区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 根据IoU计算置信度阈值
            conf_thresh = (1 + ovr) / 2

            # 找到满足以下条件的index:
            # 1. IoU小于或等于iou_thresh
            # 2. 或者置信度大于等于计算得到的conf_thresh
            inds = np.where((ovr <= iou_thresh) | (scores[order[1:]] >= conf_thresh))[0]

            # 更新order数组
            order = order[inds + 1]

        return keep

    else:
        ids1 = filtered_boxes_image1[:, 0]
        ids2 = filtered_boxes_image2[:, 0]
        bbox1 = filtered_boxes_image1[np.isin(ids1, ids2)]
        bbox1_indices = boxes_image1[np.isin(boxes_image1[:, 0], bbox1[:, 0])]

        # 获取 bbox1_indices 的 bounding boxes
        bbox1_boxes = bbox1_indices[:, :6]  # 保留 [ID, xmin, ymin, xmax, ymax, confidence]

        # 从 boxes_image1 中排除 bbox1_indices 自身，并保留 [ID, xmin, ymin, xmax, ymax, confidence]
        non_bbox1_boxes = boxes_image1[~np.isin(boxes_image1[:, 0], bbox1_indices[:, 0])][:, :6]

        # 找到 bbox1_indices 中每个框与 boxes_image1 中除自身外重叠度最大的框
        max_iou_results = []

        for i, bbox1 in enumerate(bbox1_boxes):
            max_iou = 0
            max_iou_box = None  # 用来存储 IoU 最大的非 bbox1 框
            for j, other_box in enumerate(non_bbox1_boxes):
                iou = compute_iou(bbox1, other_box)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_box = other_box

            # 只有在找到重叠框的情况下才记录结果
            if max_iou_box is not None:
                max_iou_results.append((bbox1[0], max_iou_box[0], max_iou, max_iou_box))
            else:
                # 如果没有找到符合条件的最大 IoU 框，使用默认值或跳过
                max_iou_results.append((bbox1[0], None, max_iou, None))  # 保留 None 值，确保代码后续处理


        x1 = boxes_image1[:, 1]  # xmin
        y1 = boxes_image1[:, 2]  # ymin
        x2 = boxes_image1[:, 3]  # xmax
        y2 = boxes_image1[:, 4]  # ymax
        scores = boxes_image1[:, 5]  # confidence
        # 提高 bbox1_indices 对应框的置信度
        #print("bbox1_indices", bbox1_indices)

        # 调整 bbox1_indices 的置信度：如果 bbox1 的置信度低于其重叠度最大的框的置信度，且满足新条件
        for i, bbox1 in enumerate(bbox1_boxes):
            max_iou_box = max_iou_results[i][3]  # 获取与 bbox1 重叠度最大的框信息

            # 条件1: bbox1 的置信度在 0.3 到 0.5 之间
            if 0.2 <= bbox1[5] <= 0.9   :
                # 条件2: bbox1 与周围框的最大 IoU 大于 0.2
                if max_iou_results[i][2] > 0.2:  # max_iou_results[i][2] 存储 bbox1 与其他框的最大 IoU

                    if bbox1[5] < max_iou_box[5]:  # 如果 bbox1 的置信度小于 IoU 最大框的置信度
                        # 获取 bbox1 在 boxes_image1 中的索引位置
                        index_in_boxes_image1 = np.where(boxes_image1[:, 0] == bbox1[0])[0]

                        if len(index_in_boxes_image1) > 0:
                            # 更新置信度
                            scores[index_in_boxes_image1[0]] = max_iou_box[5] - 0.01  # 调整 bbox1 的置信度为 IoU 最大框的置信度


        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            # 选择大于x1,y1和小于x2,y2的区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 根据IoU计算置信度阈值
            conf_thresh = (1 + ovr) / 2

            # 找到满足以下条件的index:
            # 1. IoU小于或等于iou_thresh
            # 2. 或者置信度大于等于计算得到的conf_thresh
            inds = np.where((ovr <= iou_thresh) | (scores[order[1:]] >= conf_thresh))[0]

            # 更新order数组
            order = order[inds + 1]


        return keep
'''
'''
def all_nms_plus_1(image1, image2, dets, dets2, homography_matrix, iou_thresh,method=1):
    boxes_image1 = dets
    boxes_image2 = dets2
    k1 = 0.6
    k2 = 0.9
    overlap_mask = find_overlap_region(image1, image2, homography_matrix)
    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, boxes_image2,
                                                                                  homography_matrix, overlap_mask)


    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 1]  # xmin
        y1 = boxes_image1[:, 2]  # ymin
        x2 = boxes_image1[:, 3]  # xmax
        y2 = boxes_image1[:, 4]  # ymax
        scores = boxes_image1[:, 5]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序


        for i in range(len(order)):

            idx = order[i]

            tx1, ty1, tx2, ty2, ts = x1[idx], y1[idx], x2[idx], y2[idx], scores[idx]

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(tx1, x1[order[i + 1:]])
            yy1 = np.maximum(ty1, y1[order[i + 1:]])
            xx2 = np.minimum(tx2, x2[order[i + 1:]])
            yy2 = np.minimum(ty2, y2[order[i + 1:]])

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 计算iou
            ovr = inter / (areas[idx] + areas[order[i + 1:]] - inter)

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            conf_thresh[ovr > iou_thresh] = 0.8 * ovr[ovr > iou_thresh] + iou_thresh

            # 使用高斯衰减函数或线性衰减函数
            if method == 1:
                weight = np.maximum(0.0, 1 - ovr)
            elif method == 2:
                weight = np.exp(-(ovr * ovr) / sigma)
            else:
                raise ValueError("method must be 1 (linear) or 2 (gaussian)")

            # 找到满足 IoU 大于 iou_thresh 且置信度小于或等于 conf_thresh 的索引
            decay_mask = (ovr > iou_thresh) & (scores[order[i + 1:]] <= conf_thresh)

            if decay_mask.any():
                # 对满足条件的项进行置信度衰减
                scores[order[i + 1:]][decay_mask] *= weight[decay_mask]

        # 筛选出大于阈值的detection
        keep_boxes = boxes_image1[scores > 0.2]
        return keep_boxes

    else:
        filtered_boxes_image_big, filtered_boxes_image_small, max_area, min_area, M = find_big_and_small_detection_boxes_middle(
            filtered_boxes_image1)

        ids1 = filtered_boxes_image_big[:, 0]
        ids3 = filtered_boxes_image_small[:, 0]
        ids2 = filtered_boxes_image2[:, 0]

        bbox1 = filtered_boxes_image_big[~np.isin(ids1, ids2)]
        bbox2 = filtered_boxes_image_small[~np.isin(ids3, ids2)]

        # 当没有大框或小框匹配时分别处理
        if bbox1.shape[0] == 0 and bbox2.shape[0] == 0:
            print("no matched boxes found")
            #return boxes_image1, bboxes_image_f
        elif bbox1.shape[0] == 0:
            print("No large boxes matched")
            id_bbox2 = bbox2[:, 0]
            id_boxes_image1 = boxes_image1[:, 0]
            matching_indices2 = np.isin(id_boxes_image1, id_bbox2)
            boxes_image1[matching_indices2, 5] *= k2
            #bboxes_image_f = boxes_image1[matching_indices2]
        elif bbox2.shape[0] == 0:
            print("No small boxes matched")
            id_bbox1 = bbox1[:, 0]
            id_boxes_image1 = boxes_image1[:, 0]
            matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
            boxes_image1[matching_indices1, 5] *= k1
            #bboxes_image_f = boxes_image1[matching_indices1]
        else:
            # 如果同时匹配到大框和小框，分别处理
            id_bbox1 = bbox1[:, 0]
            id_bbox2 = bbox2[:, 0]

            id_boxes_image1 = boxes_image1[:, 0]

            matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
            matching_indices2 = np.isin(id_boxes_image1, id_bbox2)

            # 只缩放置信度
            boxes_image1[matching_indices1, 5] *= k1
            boxes_image1[matching_indices2, 5] *= k2

            # 合并大框和小框的检测框
            #bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        kk = 0.5
        x1 = boxes_image1[:, 1]  # xmin
        y1 = boxes_image1[:, 2]  # ymin
        x2 = boxes_image1[:, 3]  # xmax
        y2 = boxes_image1[:, 4]  # ymax
        scores = boxes_image1[:, 5]  # confidence
        #scores = kk * scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序

        for i in range(len(order)):

            idx = order[i]

            tx1, ty1, tx2, ty2, ts = x1[idx], y1[idx], x2[idx], y2[idx], scores[idx]

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(tx1, x1[order[i + 1:]])
            yy1 = np.maximum(ty1, y1[order[i + 1:]])
            xx2 = np.minimum(tx2, x2[order[i + 1:]])
            yy2 = np.minimum(ty2, y2[order[i + 1:]])

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 计算iou
            ovr = inter / (areas[idx] + areas[order[i + 1:]] - inter)

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            conf_thresh[ovr > iou_thresh] = 0.8 * ovr[ovr > iou_thresh] + iou_thresh

            # 使用高斯衰减函数或线性衰减函数
            if method == 1:
                weight = np.maximum(0.0, 1 - ovr)
            elif method == 2:
                weight = np.exp(-(ovr * ovr) / sigma)
            else:
                raise ValueError("method must be 1 (linear) or 2 (gaussian)")

            # 找到满足 IoU 大于 iou_thresh 且置信度小于或等于 conf_thresh 的索引
            decay_mask = (ovr > iou_thresh) & (scores[order[i + 1:]] <= conf_thresh)

            if decay_mask.any():
                # 对满足条件的项进行置信度衰减
                scores[order[i + 1:]][decay_mask] *= weight[decay_mask]

        # 筛选出大于阈值的detection
        keep_boxes = boxes_image1[scores > 0.2]
        return keep_boxes
'''

def all_nms_plus_1(image1, image2, dets, dets2, homography_matrix, iou_thresh):
    boxes_image1 = dets.copy()
    boxes_image2 = dets2.copy()
    k1 = 0.6
    k2 = 0.9
    overlap_mask = find_overlap_region(image1, image2, homography_matrix)
    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region1(boxes_image1, boxes_image2,
                                                                                  homography_matrix, overlap_mask)


    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 1]  # xmin
        y1 = boxes_image1[:, 2]  # ymin
        x2 = boxes_image1[:, 3]  # xmax
        y2 = boxes_image1[:, 4]  # ymax
        scores = boxes_image1[:, 5]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        return keep

    else:
        filtered_boxes_image_big, filtered_boxes_image_small, max_area, min_area, M = find_big_and_small_detection_boxes_middle(
            filtered_boxes_image1)

        ids1 = filtered_boxes_image_big[:, 0]
        ids3 = filtered_boxes_image_small[:, 0]
        ids2 = filtered_boxes_image2[:, 0]

        bbox1 = filtered_boxes_image_big[~np.isin(ids1, ids2)]
        bbox2 = filtered_boxes_image_small[~np.isin(ids3, ids2)]

        # 当没有大框或小框匹配时分别处理
        if bbox1.shape[0] == 0 and bbox2.shape[0] == 0:
            print("no matched boxes found")
            #return boxes_image1, bboxes_image_f
        elif bbox1.shape[0] == 0:
            print("No large boxes matched")
            id_bbox2 = bbox2[:, 0]
            id_boxes_image1 = boxes_image1[:, 0]
            matching_indices2 = np.isin(id_boxes_image1, id_bbox2)
            boxes_image1[matching_indices2, 5] *= k2
            #bboxes_image_f = boxes_image1[matching_indices2]
        elif bbox2.shape[0] == 0:
            print("No small boxes matched")
            id_bbox1 = bbox1[:, 0]
            id_boxes_image1 = boxes_image1[:, 0]
            matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
            boxes_image1[matching_indices1, 5] *= k1
            #bboxes_image_f = boxes_image1[matching_indices1]
        else:
            # 如果同时匹配到大框和小框，分别处理
            id_bbox1 = bbox1[:, 0]
            id_bbox2 = bbox2[:, 0]

            id_boxes_image1 = boxes_image1[:, 0]

            matching_indices1 = np.isin(id_boxes_image1, id_bbox1)
            matching_indices2 = np.isin(id_boxes_image1, id_bbox2)

            # 只缩放置信度
            boxes_image1[matching_indices1, 5] *= k1
            boxes_image1[matching_indices2, 5] *= k2

            # 合并大框和小框的检测框
            #bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        kk = 0.5
        x1 = boxes_image1[:, 1]  # xmin
        y1 = boxes_image1[:, 2]  # ymin
        x2 = boxes_image1[:, 3]  # xmax
        y2 = boxes_image1[:, 4]  # ymax
        scores = boxes_image1[:, 5]  # confidence
        #scores = kk * scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 只对 IoU 大于阈值的框进行加权
            #inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            #scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
                #inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        # 获取两个 boxes 数组中的 ID
        #id_11 = keep[:, 0]  # boxes_image1 中的 ID
        #id_22 = dets[:, 0]  # boxes_image1_ori 中的 ID

        # 找到 boxes_image1 中与 boxes_image1_ori 中 ID 相同的框的索引
        #matching_indices = np.isin(id_11, id_22)

        # 获取 boxes_image1 中匹配的框的 ID
        #matching_ids = id_11[matching_indices]

        # 查找 boxes_image1_ori 中对应匹配的框
        #matching_boxes_ori = np.array(
            #[dets[dets[:, 0] == match_id] for match_id in matching_ids]).squeeze()

        # 将 boxes_image1 中匹配的框替换为 boxes_image1_ori 中对应的框
        #keep[matching_indices] = matching_boxes_ori

        return keep

def mve_nms(image1, image2, dets, dets2, homography_matrix, iou_thresh, thre, k1, k2,model,frame_id,ids1,labels1,max_id,bboxes1):
    boxes_image1 = dets.copy()
    boxes_image2 = dets2.copy()


    overlap_mask, overlap_region,x_offset, y_offset,area_ratio = find_overlap_region(image1, image2, homography_matrix)
    print("kllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk    area_ratio:", area_ratio)  # 输出面积比
    bboxes_in_overmask = np.array([])
    if overlap_region is not None:
        # 在 overlap_region 上调用模型进行检测
        bboxes_in_overmask = detect_on_overlap_region(
            model,
            "/home/hyh/PycharmProjects/mdmt-code/demo/image_1.jpg",
            x_offset=x_offset,
            y_offset=y_offset,
            frame_id=frame_id,
            bboxes1=bboxes1,
            ids1=ids1,
            labels1=labels1,
            max_id=max_id
        )
    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region_det(boxes_image1, boxes_image2,
                                                                                      homography_matrix, overlap_mask)

    if area_ratio > 0.95 or bboxes_in_overmask.shape[0] == 0 or filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box
        change_det1 = np.array([])

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i], areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        return keep, change_det1

    else:

        M = find_big_and_small_detection_boxes_middle_det(filtered_boxes_image1)

        boxes_image1 = boxes_image1[boxes_image1[:, 4].argsort()[::-1]]
        area = (boxes_image1[:, 3] - boxes_image1[:, 1]) * (boxes_image1[:, 2] - boxes_image1[:, 0])

        cent_1x = (boxes_image1[:, 2] + boxes_image1[:, 0]) / 2
        cent_1y = (boxes_image1[:, 3] + boxes_image1[:, 1]) / 2

        cent_1 = np.stack([cent_1x, cent_1y], axis=1)
        top_k = 5
        # 初始化索引映射
        cent_1_indices = np.arange(len(cent_1))

        # 初始化变量
        unmatched_dets = []
        unmatched_dets2 = []
        matched_dets = []
        matched_dets2 = []
        change_det_list = []

        # 遍历 boxes_image2 的每个检测框
        for i, det in enumerate(boxes_image2):
            if len(cent_1) == 0:
                break

            x1, y1, x2, y2, score = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)

            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0, 0], transformed_point[0, 0, 1]

            distances = np.sqrt((transformed_x - cent_1[:, 0]) ** 2 + (transformed_y - cent_1[:, 1]) ** 2)
            min_distance = np.min(distances)
            min_index = np.argmin(distances)

            if min_distance < thre:
                matched_index = cent_1_indices[min_index]
                matched_dets.append(boxes_image1[matched_index])
                matched_dets2.append(det)
                cent_1 = np.delete(cent_1, min_index, axis=0)
                cent_1_indices = np.delete(cent_1_indices, min_index)
            else:
                unmatched_dets2.append(det)

        # 未匹配的检测框
        new_boxes = []
        unmatched_dets = boxes_image1[cent_1_indices]
        unmatched_indices = cent_1_indices

        # 确保 unmatched_dets2 和 bboxes_in_overmask 都是二维数组
        unmatched_dets2 = np.array(unmatched_dets2).reshape(-1, 5)
        bboxes_in_overmask = np.array(bboxes_in_overmask).reshape(-1, 5)
        boxes_image1 = np.array(boxes_image1).reshape(-1, 5)

        # 步骤 1：按置信度排序，选择置信度最高的 5 个检测框
        unmatched_dets2 = unmatched_dets2[unmatched_dets2[:, 4].argsort()[::-1]]
        top5_dets2 = unmatched_dets2[:5]

        # 步骤 2：将中心点映射到 image1 的坐标系
        mapped_centers = []
        for det in top5_dets2:
            x1, y1, x2, y2, score = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0]
            mapped_centers.append((transformed_x, transformed_y))

        # 步骤 3：计算 bboxes_in_overmask 中检测框的中心点
        bboxes_centers = []
        for bbox in bboxes_in_overmask:
            x1, y1, x2, y2, score = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bboxes_centers.append((center_x, center_y))
        bboxes_centers = np.array(bboxes_centers)

        # 计算 boxes_image1 中现有检测框的中心点
        boxes_image1_centers = []
        for bbox in boxes_image1:
            x1, y1, x2, y2, score = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            boxes_image1_centers.append((center_x, center_y))
        boxes_image1_centers = np.array(boxes_image1_centers)

        # 步骤 4：比较中心点距离并更新 boxes_image1
        distance_threshold = 100

        for idx_mapped, mapped_center in enumerate(mapped_centers):
            transformed_x, transformed_y = mapped_center
            # 计算 mapped_center 到 bboxes_centers 所有中心点的距离
            distances = np.sqrt(
                (transformed_x - bboxes_centers[:, 0]) ** 2 + (transformed_y - bboxes_centers[:, 1]) ** 2)
            idx = np.argmin(distances)
            min_distance = distances[idx]

            if min_distance < distance_threshold:
                # 筛选出与 mapped_center 距离小于 50 的检测框
                close_indices = np.where(distances < distance_threshold)[0]

                if len(close_indices) > 0:
                    # 从 close_indices 中选出置信度最高的检测框
                    highest_confidence_index = max(close_indices, key=lambda i: bboxes_in_overmask[i, 4])
                    box_to_add = bboxes_in_overmask[highest_confidence_index]
                    if box_to_add[4] > 0.5:

                        # 计算 box_to_add 与 boxes_image1 中现有框的距离
                        distances_to_boxes_image1 = np.sqrt(
                            ((box_to_add[0] + box_to_add[2]) / 2 - boxes_image1_centers[:, 0]) ** 2 +
                            ((box_to_add[1] + box_to_add[3]) / 2 - boxes_image1_centers[:, 1]) ** 2
                        )
                        min_distance_to_boxes_image1 = np.min(distances_to_boxes_image1)

                        # 只有当最小距离大于 50 时才添加
                        if min_distance_to_boxes_image1 > 50:
                            # 计算与 boxes_image1 中所有框的 IoU
                            ious = [calculate_iou(box_to_add, box) for box in boxes_image1]

                            # 仅当所有 IoU 都小于 0.2 时才添加
                            if all(iou < 0.2 for iou in ious):
                                if not np.any(np.all(boxes_image1 == box_to_add, axis=1)):
                                    boxes_image1 = np.vstack((boxes_image1, box_to_add))
                                    new_boxes.append(box_to_add)
                                    # 更新 area 和其他相关变量
                                    new_area = (box_to_add[3] - box_to_add[1]) * (box_to_add[2] - box_to_add[0])
                                    area = np.append(area, new_area)

        #将 new_boxes 转换为 numpy 数组
        new_boxes = np.array(new_boxes)

        # 处理未匹配的检测框
        if len(unmatched_dets) > 8:
            change_det = np.array([])
        else:
            processed_indices = []

            # 第一次循环
            for idx, det in enumerate(unmatched_dets):
                min_index = unmatched_indices[idx]
                if det[4] < 0.5 and area[min_index] > M:
                    change_det_list.append(det)
                    boxes_image1[min_index, 4] *= k1
                    processed_indices.append(idx)

            # 第二次循环
            top_k = 5
            unprocessed_indices = [i for i in range(len(unmatched_dets)) if i not in processed_indices]
            unprocessed_dets = unmatched_dets[unprocessed_indices]
            sorted_indices = np.argsort(unprocessed_dets[:, 4])[::-1]
            processed_count = 0

            for idx in sorted_indices:
                if processed_count >= top_k:
                    break
                det = unprocessed_dets[idx]
                min_index = unmatched_indices[unprocessed_indices[idx]]
                processed_count += 1
                change_det_list.append(det)
                if area[min_index] > M:
                    boxes_image1[min_index, 4] *= k1
                else:
                    boxes_image1[min_index, 4] *= k2
                processed_indices.append(unprocessed_indices[idx])

            # 将收集的检测框转换为数组
            change_det = np.array(change_det_list) if change_det_list else np.array([])

        # 合并大框和小框的检测框
        # bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        kk = 0.5
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence
        # scores = kk * scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i],areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            # conf_thresh = np.ones_like(ovr) * 0.2
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        for i, box_keep in enumerate(keep):
            # 获取keep中当前检测框的坐标 (x1, y1, x2, y2)
            x1_k, y1_k, x2_k, y2_k = box_keep[:4]

            # 遍历det1中的每个检测框，找出坐标相同的检测框
            for box_det in dets:

                x1_d, y1_d, x2_d, y2_d, score_d = box_det

                # 如果坐标完全相等
                if x1_k == x1_d and y1_k == y1_d and x2_k == x2_d and y2_k == y2_d:
                    # 替换keep中检测框的置信度为det1中对应的置信度
                    keep[i, 4] = score_d
        # 只比较坐标部分，忽略分数
        coords_dets1 = dets[:, :4]
        coords_boxes_image1 = keep[:, :4]

        # 找到在 dets1 中但不在 boxes_image1 中的检测框（只比较坐标）
        unique_coords = []
        for coord in coords_dets1:
            if not any(np.array_equal(coord, box_coord) for box_coord in coords_boxes_image1):
                unique_coords.append(coord)

        # 根据找到的唯一坐标从 dets1 中提取完整的检测框
        unique_dets = np.array([d for d in dets if any(np.array_equal(d[:4], uc) for uc in unique_coords)])

        return keep, new_boxes

def mve_nms1(img, image1, image2, dets, dets2, homography_matrix, iou_thresh, thre, k1, k2,detector,thret):
    boxes_image1 = dets.copy()
    boxes_image2 = dets2.copy()

    sub_images_info = split_image_single_folder(img, "/home/ubuntu/PycharmProject/Yolov7_StrongSORT_OSNet-main/image_1/",2,2)

    overlap_mask, overlap_region,x_offset, y_offset,area_ratio = find_overlap_region(image1, image2, homography_matrix)
    print("kllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk    area_ratio:", area_ratio)  # 输出面积比
    #bboxes_in_overmask = np.array([])

    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region_det_cen(boxes_image1, boxes_image2,
                                                                                      homography_matrix, overlap_mask)
    print("filtered_boxes_image1", filtered_boxes_image1.size)
    print("filtered_boxes_image2", filtered_boxes_image2.size)

    if  filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box
        change_det1 = np.array([])
        det_delete = np.array([])
        """00
        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i], areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        return keep, change_det1, det_delete
        """
        return dets, change_det1, det_delete,keep

    else:

        M = find_big_and_small_detection_boxes_middle_det(filtered_boxes_image1)

        filtered_boxes_image1 = filtered_boxes_image1[filtered_boxes_image1[:, 4].argsort()[::-1]]
        filtered_boxes_image2 = filtered_boxes_image2[filtered_boxes_image2[:, 4].argsort()[::-1]]
        area = (filtered_boxes_image1[:, 3] - filtered_boxes_image1[:, 1]) * (filtered_boxes_image1[:, 2] - filtered_boxes_image1[:, 0])

        cent_1x = (filtered_boxes_image1[:, 2] + filtered_boxes_image1[:, 0]) / 2
        cent_1y = (filtered_boxes_image1[:, 3] + filtered_boxes_image1[:, 1]) / 2

        cent_1 = np.stack([cent_1x, cent_1y], axis=1)
        top_k = 5
        # 初始化索引映射
        cent_1_indices = np.arange(len(cent_1))

        # 初始化变量
        unmatched_dets = []
        unmatched_dets2 = []
        matched_dets = []
        matched_dets2 = []
        change_det_list = []

        # 遍历 boxes_image2 的每个检测框
        for i, det in enumerate(filtered_boxes_image2):
            if len(cent_1) == 0:
                break

            x1, y1, x2, y2, score, cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)

            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0, 0], transformed_point[0, 0, 1]

            distances = np.sqrt((transformed_x - cent_1[:, 0]) ** 2 + (transformed_y - cent_1[:, 1]) ** 2)
            min_distance = np.min(distances)
            min_index = np.argmin(distances)

            if min_distance < thre:
                matched_index = cent_1_indices[min_index]
                matched_dets.append(filtered_boxes_image1[matched_index])
                matched_dets2.append(det)
                cent_1 = np.delete(cent_1, min_index, axis=0)
                cent_1_indices = np.delete(cent_1_indices, min_index)
            else:
                unmatched_dets2.append(det)

        # 未匹配的检测框
        new_boxes = []
        unmatched_dets = filtered_boxes_image1[cent_1_indices]
        unmatched_indices = cent_1_indices

        # 确保 unmatched_dets2 和 bboxes_in_overmask 都是二维数组
        unmatched_dets2 = np.array(unmatched_dets2).reshape(-1, 6)
        #bboxes_in_overmask = np.array(bboxes_in_overmask).reshape(-1, 5)
        boxes_image1 = np.array(boxes_image1).reshape(-1, 6)

        # 步骤 1：按置信度排序，选择置信度最高的 5 个检测框
        unmatched_dets2 = unmatched_dets2[unmatched_dets2[:, 4].argsort()[::-1]]
        #top5_dets2 = unmatched_dets2[:5]
        #print("pppppppppppppppppppppppppppppppp",unmatched_dets2.shape[0])
        top = min(10,len(unmatched_dets2))
        top5_dets2 = unmatched_dets2[:top]
        #top5_dets2 = unmatched_dets2

        # 步骤 2：将中心点映射到 image1 的坐标系
        mapped_centers = []
        for det in top5_dets2:
            x1, y1, x2, y2, score,cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0]
            mapped_centers.append((transformed_x, transformed_y))

        processed_ids = set()
        adjusted_bboxes = []
        adjusted_bboxes2 = []
        before_bboxes = []

        for center_x, center_y in mapped_centers:
            # 找到包含该中心点的子图
            for sub_info in sub_images_info:
                left, upper, right, lower, sub_id = sub_info['left'], sub_info['upper'], sub_info['right'], sub_info[
                    'lower'], sub_info['id']

                if left <= center_x < right and upper <= center_y < lower:
                    # 检查该子图的 id 是否已被处理
                    if sub_id in processed_ids:
                        print(f"子图像 ID {sub_id} 已被处理，跳过检测。")
                        break  # 跳过当前子图

                    # 标记该子图为已处理
                    processed_ids.add(sub_id)

                    # 子图区域对应的文件路径
                    overlap_region_path = sub_info['filename']
                    image_p = cv2.imread(overlap_region_path)

                    im0,det_bboxes_array = detector.detect(image_p)
                    ddet_before = det_bboxes_array

                    iou_threshold = 0.3
                    if len(det_bboxes_array) > 0:
                        # 执行 NMS
                        indices = cv2.dnn.NMSBoxes(
                            bboxes=det_bboxes_array[:, :4].tolist(),
                            scores=det_bboxes_array[:, 4].tolist(),
                            score_threshold=0.1,  # 设置得分阈值，过滤低置信度检测
                            nms_threshold=iou_threshold
                        )
                        print("NMS indices:", indices)  # 输出 indices 以便检查

                        # 根据 NMS 结果过滤 det_bboxes_array
                        # 直接遍历 indices 中的每个索引 i
                        nms_filtered_detections = np.array([det_bboxes_array[i] for i in indices])

                    # 检查 result['det_bboxes'] 的结构
                    #det_bboxes_array = result.get('det_bboxes', [])  # 使用 get 以防键不存在
                    distance_threshold = 50
                    if len(det_bboxes_array) > 0:
                        #det_bboxes_array = det_bboxes_array[0]  # 取出第一个元素，假设是 [x_min, y_min, x_max, y_max, score]
                        edge_threshold = 10
                        # 获取子图的位置信息
                        x_offset = left
                        y_offset = upper

                        for bbox in ddet_before:
                            x_min, y_min, x_max, y_max, score, lbl = bbox

                            # 计算检测框的中心点
                            #bbox_center_x = (x_min + x_max) / 2
                            #bbox_center_y = (y_min + y_max) / 2

                            # 计算中心点之间的距离
                            #distance = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)
                            adjusted_bbox_be = [
                                x_min + x_offset,
                                y_min + y_offset,
                                x_max + x_offset,
                                y_max + y_offset,
                                score,  # 保留置信度分数
                                lbl
                            ]
                            before_bboxes.append(adjusted_bbox_be)

                        # 迭代每个 NMS 过滤后的检测框
                        for bbox in nms_filtered_detections:
                            x_min, y_min, x_max, y_max, score, lbl = bbox

                            # 计算检测框的中心点
                            #bbox_center_x = (x_min + x_max) / 2
                            #bbox_center_y = (y_min + y_max) / 2

                            # 计算中心点之间的距离
                            #distance = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)
                            adjusted_bbox = [
                                x_min + x_offset,
                                y_min + y_offset,
                                x_max + x_offset,
                                y_max + y_offset,
                                score,  # 保留置信度分数
                                lbl
                            ]
                            adjusted_bboxes2.append(adjusted_bbox)

                            # 检查是否满足距离和边缘条件
                            if (
                                    x_min >= edge_threshold and
                                    y_min >= edge_threshold and
                                    x_max <= 960 - edge_threshold and
                                    y_max <= 540 - edge_threshold):
                                # 将边界框坐标加上偏移量 (x_offset, y_offset)
                                adjusted_bbox = [
                                    x_min + x_offset,
                                    y_min + y_offset,
                                    x_max + x_offset,
                                    y_max + y_offset,
                                    score,  # 保留置信度分数
                                    lbl
                                ]
                                #adjusted_bboxes2.append(adjusted_bbox)
                                if score > thret:
                                    adjusted_bboxes.append(adjusted_bbox)

                    break  # 找到对应的子图后，跳出子图循环

        adjusted_bboxes = np.array(adjusted_bboxes)
        adjusted_bboxes2 = np.array(adjusted_bboxes2)
        before_bboxes = np.array(before_bboxes)
        #print("pppppppppppppppppppppppppppppppppp", adjusted_bboxes)
        #print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm", processed_ids)
        """
        # 将 sub_images_info 转换为按 id 索引的字典
        sub_images_dict = {sub_img['id']: sub_img for sub_img in sub_images_info}

        # 筛选符合条件的检测框
        filtered_dets = []

        for det in unmatched_dets:
            x1, y1, x2, y2, confidence, category = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            sub_id = get_sub_image_id_for_point(center_x, center_y, sub_images_dict)
            if sub_id is not None and sub_id in processed_ids:
                filtered_dets.append(det)

        # 确保 boxes_image1 是一个 NumPy 数组
        boxes_image1 = np.array(boxes_image1)
        filtered_dets = np.array(filtered_dets)
        # 初始化一个列表来存储需要添加的新检测框
        new_boxes = []
        repress_boxes = []
        

        for i,existing_bbox in enumerate(filtered_dets):
            # 初始化一个标志，表示是否可以添加该检测框
            can_repre = True

            # 遍历 boxes_image1 中的每个检测框
            for adjusted_bbox in adjusted_bboxes2:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                #intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                #adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                #overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.5 :
                    can_repre = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_repre:
                matches = np.where(
                    (boxes_image1[:, 0] == existing_bbox[0]) &
                    (boxes_image1[:, 1] == existing_bbox[1]) &
                    (boxes_image1[:, 2] == existing_bbox[2]) &
                    (boxes_image1[:, 3] == existing_bbox[3])
                )[0]
                boxes_image1[matches,4] *= 0.75
                repress_boxes.append(existing_bbox)
        if repress_boxes:
            repress_boxes = np.array(repress_boxes)
        """
        # 遍历 adjusted_bboxes 中的每个检测框
        for adjusted_bbox in adjusted_bboxes:
            # 初始化一个标志，表示是否可以添加该检测框
            can_add = True

            # 遍历 boxes_image1 中的每个检测框
            for existing_bbox in boxes_image1:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.2 or overlap_ratio > 0.5:
                    can_add = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_add:
                new_boxes.append(adjusted_bbox)

        # 将 new_boxes 中的检测框添加到 boxes_image1 中
        if new_boxes:
            new_boxes = np.array(new_boxes)
            boxes_image1 = np.vstack((boxes_image1, new_boxes))

        #将 new_boxes 转换为 numpy 数组
        new_boxes = np.array(new_boxes)
        unique_dets = np.array([])
        """
        # 处理未匹配的检测框
        if len(unmatched_dets) > 8:
            change_det = np.array([])
        else:
            processed_indices = []

            # 第一次循环
            for idx, det in enumerate(unmatched_dets):
                min_index = unmatched_indices[idx]
                if det[4] < 0.5 and area[min_index] > M:
                    change_det_list.append(det)
                    boxes_image1[min_index, 4] *= k1
                    processed_indices.append(idx)

            # 第二次循环
            top_k = 5
            unprocessed_indices = [i for i in range(len(unmatched_dets)) if i not in processed_indices]
            unprocessed_dets = unmatched_dets[unprocessed_indices]
            sorted_indices = np.argsort(unprocessed_dets[:, 4])[::-1]
            processed_count = 0

            for idx in sorted_indices:
                if processed_count >= top_k:
                    break
                det = unprocessed_dets[idx]
                min_index = unmatched_indices[unprocessed_indices[idx]]
                processed_count += 1
                change_det_list.append(det)
                if area[min_index] > M:
                    boxes_image1[min_index, 4] *= k1
                else:
                    boxes_image1[min_index, 4] *= k2
                processed_indices.append(unprocessed_indices[idx])

            # 将收集的检测框转换为数组
            change_det = np.array(change_det_list) if change_det_list else np.array([])

        # 合并大框和小框的检测框
        # bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        kk = 0.5
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence
        # scores = kk * scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标


            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            #yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i],areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            # conf_thresh = np.ones_like(ovr) * 0.2
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        for i, box_keep in enumerate(keep):
            # 获取keep中当前检测框的坐标 (x1, y1, x2, y2)
            x1_k, y1_k, x2_k, y2_k = box_keep[:4]

            # 遍历det1中的每个检测框，找出坐标相同的检测框
            for box_det in dets:

                x1_d, y1_d, x2_d, y2_d, score_d, lbll = box_det

                # 如果坐标完全相等
                if x1_k == x1_d and y1_k == y1_d and x2_k == x2_d and y2_k == y2_d:
                    # 替换keep中检测框的置信度为det1中对应的置信度
                    keep[i, 4] = score_d
        # 只比较坐标部分，忽略分数
        coords_dets1 = dets[:, :4]
        coords_boxes_image1 = keep[:, :4]

        # 找到在 dets1 中但不在 boxes_image1 中的检测框（只比较坐标）
        unique_coords = []
        for coord in coords_dets1:
            if not any(np.array_equal(coord, box_coord) for box_coord in coords_boxes_image1):
                unique_coords.append(coord)

        # 根据找到的唯一坐标从 dets1 中提取完整的检测框
        unique_dets = np.array([d for d in dets if any(np.array_equal(d[:4], uc) for uc in unique_coords)])

        return keep, new_boxes, unique_dets
        """
        return boxes_image1, new_boxes, before_bboxes,adjusted_bboxes2


def mve_nms1_2(img, image1, image2, dets, dets2, homography_matrix, iou_thresh, thre, k1, k2, detector, thret):
    boxes_image1 = dets.copy()
    boxes_image2 = dets2.copy()

    sub_images_info = split_image_single_folder(img,
                                                "/home/ubuntu/PycharmProject/Yolov7_StrongSORT_OSNet-main/image_1/", 2,
                                                2)

    overlap_mask, overlap_region, x_offset, y_offset, area_ratio = find_overlap_region(image1, image2,
                                                                                       homography_matrix)
    print("kllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk    area_ratio:", area_ratio)  # 输出面积比
    # bboxes_in_overmask = np.array([])

    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region_det_cen(boxes_image1, boxes_image2,
                                                                                          homography_matrix,
                                                                                          overlap_mask)
    print("filtered_boxes_image1", filtered_boxes_image1.size)
    print("filtered_boxes_image2", filtered_boxes_image2.size)

    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box
        change_det1 = np.array([])
        det_delete = np.array([])
        """00
        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i], areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        return keep, change_det1, det_delete
        """
        return dets, change_det1, det_delete, keep

    else:

        M = find_big_and_small_detection_boxes_middle_det(filtered_boxes_image1)

        filtered_boxes_image1 = filtered_boxes_image1[filtered_boxes_image1[:, 4].argsort()[::-1]]
        boxes_image2 = boxes_image2[boxes_image2[:, 4].argsort()[::-1]]
        area = (filtered_boxes_image1[:, 3] - filtered_boxes_image1[:, 1]) * (
                    filtered_boxes_image1[:, 2] - filtered_boxes_image1[:, 0])

        cent_1x = (filtered_boxes_image1[:, 2] + filtered_boxes_image1[:, 0]) / 2
        cent_1y = (filtered_boxes_image1[:, 3] + filtered_boxes_image1[:, 1]) / 2

        cent_1 = np.stack([cent_1x, cent_1y], axis=1)
        top_k = 5
        # 初始化索引映射
        cent_1_indices = np.arange(len(cent_1))

        # 初始化变量
        unmatched_dets = []
        unmatched_dets2 = []
        matched_dets = []
        matched_dets2 = []
        change_det_list = []

        # 遍历 boxes_image2 的每个检测框
        for i, det in enumerate(boxes_image2):
            if len(cent_1) == 0:
                break

            x1, y1, x2, y2, score, cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)

            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0, 0], transformed_point[0, 0, 1]

            distances = np.sqrt((transformed_x - cent_1[:, 0]) ** 2 + (transformed_y - cent_1[:, 1]) ** 2)
            min_distance = np.min(distances)
            min_index = np.argmin(distances)

            if min_distance < thre:
                matched_index = cent_1_indices[min_index]
                matched_dets.append(filtered_boxes_image1[matched_index])
                matched_dets2.append(det)
                cent_1 = np.delete(cent_1, min_index, axis=0)
                cent_1_indices = np.delete(cent_1_indices, min_index)
            else:
                unmatched_dets2.append(det)

        # 未匹配的检测框
        new_boxes = []
        unmatched_dets = filtered_boxes_image1[cent_1_indices]
        unmatched_indices = cent_1_indices

        # 确保 unmatched_dets2 和 bboxes_in_overmask 都是二维数组
        unmatched_dets2 = np.array(unmatched_dets2).reshape(-1, 6)
        # bboxes_in_overmask = np.array(bboxes_in_overmask).reshape(-1, 5)
        boxes_image1 = np.array(boxes_image1).reshape(-1, 6)

        # 步骤 1：按置信度排序，选择置信度最高的 5 个检测框
        unmatched_dets2 = unmatched_dets2[unmatched_dets2[:, 4].argsort()[::-1]]
        # top5_dets2 = unmatched_dets2[:5]
        # print("pppppppppppppppppppppppppppppppp",unmatched_dets2.shape[0])
        top = min(10, len(unmatched_dets2))
        top5_dets2 = unmatched_dets2[:top]
        # top5_dets2 = unmatched_dets2

        # 步骤 2：将中心点映射到 image1 的坐标系
        mapped_centers = []
        for det in top5_dets2:
            x1, y1, x2, y2, score, cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0]
            mapped_centers.append((transformed_x, transformed_y))

        processed_ids = set()
        adjusted_bboxes = []
        adjusted_bboxes2 = []

        for center_x, center_y in mapped_centers:
            # 找到包含该中心点的子图
            for sub_info in sub_images_info:
                left, upper, right, lower, sub_id = sub_info['left'], sub_info['upper'], sub_info['right'], sub_info[
                    'lower'], sub_info['id']

                if left <= center_x < right and upper <= center_y < lower:
                    # 检查该子图的 id 是否已被处理
                    if sub_id in processed_ids:
                        print(f"子图像 ID {sub_id} 已被处理，跳过检测。")
                        break  # 跳过当前子图

                    # 标记该子图为已处理
                    processed_ids.add(sub_id)

                    # 子图区域对应的文件路径
                    overlap_region_path = sub_info['filename']
                    image_p = cv2.imread(overlap_region_path)

                    im0, det_bboxes_array = detector.detect(image_p)

                    iou_threshold = 0.3
                    if len(det_bboxes_array) > 0:
                        # 执行 NMS
                        indices = cv2.dnn.NMSBoxes(
                            bboxes=det_bboxes_array[:, :4].tolist(),
                            scores=det_bboxes_array[:, 4].tolist(),
                            score_threshold=0.1,  # 设置得分阈值，过滤低置信度检测
                            nms_threshold=iou_threshold
                        )
                        print("NMS indices:", indices)  # 输出 indices 以便检查

                        # 根据 NMS 结果过滤 det_bboxes_array
                        # 直接遍历 indices 中的每个索引 i
                        nms_filtered_detections = np.array([det_bboxes_array[i] for i in indices])

                    # 检查 result['det_bboxes'] 的结构
                    # det_bboxes_array = result.get('det_bboxes', [])  # 使用 get 以防键不存在
                    distance_threshold = 50
                    if len(det_bboxes_array) > 0:
                        # det_bboxes_array = det_bboxes_array[0]  # 取出第一个元素，假设是 [x_min, y_min, x_max, y_max, score]
                        edge_threshold = 10
                        # 获取子图的位置信息
                        x_offset = left
                        y_offset = upper

                        # 迭代每个 NMS 过滤后的检测框
                        for bbox in nms_filtered_detections:
                            x_min, y_min, x_max, y_max, score, lbl = bbox

                            # 计算检测框的中心点
                            # bbox_center_x = (x_min + x_max) / 2
                            # bbox_center_y = (y_min + y_max) / 2

                            # 计算中心点之间的距离
                            # distance = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)
                            adjusted_bbox = [
                                x_min + x_offset,
                                y_min + y_offset,
                                x_max + x_offset,
                                y_max + y_offset,
                                score,  # 保留置信度分数
                                lbl
                            ]
                            adjusted_bboxes2.append(adjusted_bbox)

                            # 检查是否满足距离和边缘条件
                            if (
                                    x_min >= edge_threshold and
                                    y_min >= edge_threshold and
                                    x_max <= 960 - edge_threshold and
                                    y_max <= 540 - edge_threshold):
                                # 将边界框坐标加上偏移量 (x_offset, y_offset)
                                adjusted_bbox = [
                                    x_min + x_offset,
                                    y_min + y_offset,
                                    x_max + x_offset,
                                    y_max + y_offset,
                                    score,  # 保留置信度分数
                                    lbl
                                ]
                                # adjusted_bboxes2.append(adjusted_bbox)
                                if score > thret:
                                    adjusted_bboxes.append(adjusted_bbox)

                    break  # 找到对应的子图后，跳出子图循环

        adjusted_bboxes = np.array(adjusted_bboxes)
        adjusted_bboxes2 = np.array(adjusted_bboxes2)
        # print("pppppppppppppppppppppppppppppppppp", adjusted_bboxes)
        # print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm", processed_ids)
        """
        # 将 sub_images_info 转换为按 id 索引的字典
        sub_images_dict = {sub_img['id']: sub_img for sub_img in sub_images_info}

        # 筛选符合条件的检测框
        filtered_dets = []

        for det in unmatched_dets:
            x1, y1, x2, y2, confidence, category = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            sub_id = get_sub_image_id_for_point(center_x, center_y, sub_images_dict)
            if sub_id is not None and sub_id in processed_ids:
                filtered_dets.append(det)

        # 确保 boxes_image1 是一个 NumPy 数组
        boxes_image1 = np.array(boxes_image1)
        filtered_dets = np.array(filtered_dets)
        # 初始化一个列表来存储需要添加的新检测框
        new_boxes = []
        repress_boxes = []


        for i,existing_bbox in enumerate(filtered_dets):
            # 初始化一个标志，表示是否可以添加该检测框
            can_repre = True

            # 遍历 boxes_image1 中的每个检测框
            for adjusted_bbox in adjusted_bboxes2:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                #intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                #adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                #overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.5 :
                    can_repre = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_repre:
                matches = np.where(
                    (boxes_image1[:, 0] == existing_bbox[0]) &
                    (boxes_image1[:, 1] == existing_bbox[1]) &
                    (boxes_image1[:, 2] == existing_bbox[2]) &
                    (boxes_image1[:, 3] == existing_bbox[3])
                )[0]
                boxes_image1[matches,4] *= 0.75
                repress_boxes.append(existing_bbox)
        if repress_boxes:
            repress_boxes = np.array(repress_boxes)
        """
        # 遍历 adjusted_bboxes 中的每个检测框
        for adjusted_bbox in adjusted_bboxes:
            # 初始化一个标志，表示是否可以添加该检测框
            can_add = True

            # 遍历 boxes_image1 中的每个检测框
            for existing_bbox in boxes_image1:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.2 or overlap_ratio > 0.5:
                    can_add = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_add:
                new_boxes.append(adjusted_bbox)

        # 将 new_boxes 中的检测框添加到 boxes_image1 中
        if new_boxes:
            new_boxes = np.array(new_boxes)
            boxes_image1 = np.vstack((boxes_image1, new_boxes))

        # 将 new_boxes 转换为 numpy 数组
        new_boxes = np.array(new_boxes)
        unique_dets = np.array([])
        """
        # 处理未匹配的检测框
        if len(unmatched_dets) > 8:
            change_det = np.array([])
        else:
            processed_indices = []

            # 第一次循环
            for idx, det in enumerate(unmatched_dets):
                min_index = unmatched_indices[idx]
                if det[4] < 0.5 and area[min_index] > M:
                    change_det_list.append(det)
                    boxes_image1[min_index, 4] *= k1
                    processed_indices.append(idx)

            # 第二次循环
            top_k = 5
            unprocessed_indices = [i for i in range(len(unmatched_dets)) if i not in processed_indices]
            unprocessed_dets = unmatched_dets[unprocessed_indices]
            sorted_indices = np.argsort(unprocessed_dets[:, 4])[::-1]
            processed_count = 0

            for idx in sorted_indices:
                if processed_count >= top_k:
                    break
                det = unprocessed_dets[idx]
                min_index = unmatched_indices[unprocessed_indices[idx]]
                processed_count += 1
                change_det_list.append(det)
                if area[min_index] > M:
                    boxes_image1[min_index, 4] *= k1
                else:
                    boxes_image1[min_index, 4] *= k2
                processed_indices.append(unprocessed_indices[idx])

            # 将收集的检测框转换为数组
            change_det = np.array(change_det_list) if change_det_list else np.array([])

        # 合并大框和小框的检测框
        # bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        kk = 0.5
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence
        # scores = kk * scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标


            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            #yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i],areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            # conf_thresh = np.ones_like(ovr) * 0.2
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        for i, box_keep in enumerate(keep):
            # 获取keep中当前检测框的坐标 (x1, y1, x2, y2)
            x1_k, y1_k, x2_k, y2_k = box_keep[:4]

            # 遍历det1中的每个检测框，找出坐标相同的检测框
            for box_det in dets:

                x1_d, y1_d, x2_d, y2_d, score_d, lbll = box_det

                # 如果坐标完全相等
                if x1_k == x1_d and y1_k == y1_d and x2_k == x2_d and y2_k == y2_d:
                    # 替换keep中检测框的置信度为det1中对应的置信度
                    keep[i, 4] = score_d
        # 只比较坐标部分，忽略分数
        coords_dets1 = dets[:, :4]
        coords_boxes_image1 = keep[:, :4]

        # 找到在 dets1 中但不在 boxes_image1 中的检测框（只比较坐标）
        unique_coords = []
        for coord in coords_dets1:
            if not any(np.array_equal(coord, box_coord) for box_coord in coords_boxes_image1):
                unique_coords.append(coord)

        # 根据找到的唯一坐标从 dets1 中提取完整的检测框
        unique_dets = np.array([d for d in dets if any(np.array_equal(d[:4], uc) for uc in unique_coords)])

        return keep, new_boxes, unique_dets
        """
        return boxes_image1, new_boxes, unique_dets, adjusted_bboxes2


def mve_nms1_1(img, image1, image2, dets, dets2, homography_matrix, iou_thresh, thre, k1, k2, detector, thret):
    boxes_image1 = dets.copy()
    boxes_image2 = dets2.copy()

    sub_images_info = split_image_single_folder(img,
                                                "/home/ubuntu/PycharmProject/Yolov7_StrongSORT_OSNet-main/image_1/", 2,
                                                2)

    overlap_mask, overlap_region, x_offset, y_offset, area_ratio = find_overlap_region(image1, image2,
                                                                                       homography_matrix)
    print("kllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk    area_ratio:", area_ratio)  # 输出面积比
    # bboxes_in_overmask = np.array([])

    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region_det(boxes_image1, boxes_image2,
                                                                                      homography_matrix, overlap_mask)
    print("filtered_boxes_image1", filtered_boxes_image1.size)
    print("filtered_boxes_image2", filtered_boxes_image2.size)

    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box
        change_det1 = np.array([])
        det_delete = np.array([])
        """00
        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i], areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        return keep, change_det1, det_delete
        """
        return dets, change_det1, det_delete, keep

    else:

        M = find_big_and_small_detection_boxes_middle_det(filtered_boxes_image1)

        boxes_image1 = boxes_image1[boxes_image1[:, 4].argsort()[::-1]]
        #print(type(boxes_image1))  # 确保是 list

        boxes_image2 = boxes_image2[boxes_image2[:, 4].argsort()[::-1]]
        area = (boxes_image1[:, 3] - boxes_image1[:, 1]) * (boxes_image1[:, 2] - boxes_image1[:, 0])

        cent_1x = (boxes_image1[:, 2] + boxes_image1[:, 0]) / 2
        cent_1y = (boxes_image1[:, 3] + boxes_image1[:, 1]) / 2

        cent_1 = np.stack([cent_1x, cent_1y], axis=1)
        top_k = 5
        # 初始化索引映射
        cent_1_indices = np.arange(len(cent_1))

        # 初始化变量
        unmatched_dets = []
        unmatched_dets2 = []
        matched_dets = []
        matched_dets2 = []
        change_det_list = []
        boxes_image11 = boxes_image1
        # 遍历 boxes_image2 的每个检测框
        for i, det in enumerate(boxes_image2):
            if len(boxes_image11) == 0:
                break

            # 获取检测框信息
            x1, y1, x2, y2, score, cata = det
            bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)

            # 转换检测框到另一个视角
            transformed_bbox = cv2.perspectiveTransform(bbox, homography_matrix)
            transformed_bbox = transformed_bbox.reshape(4, 2)

            # 遍历目标视角的检测框，计算IOU
            max_iou = 0
            max_iou_index = -1

            for j, bbox1 in enumerate(boxes_image11):
                x1_1, y1_1, x2_1, y2_1, _, _ = bbox1
                bbox1_coords = np.array([[x1_1, y1_1], [x2_1, y1_1], [x2_1, y2_1], [x1_1, y2_1]])

                # 计算IOU
                iou = calculate_polygon_iou(transformed_bbox, bbox1_coords)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_index = j

            # 如果IOU超过阈值，认为匹配
            if max_iou > 0.5:
                matched_dets.append(boxes_image11[max_iou_index])
                matched_dets2.append(det)
                boxes_image11 = np.delete(boxes_image11, max_iou_index, axis=0)
            else:
                unmatched_dets2.append(det)

        # 未匹配的检测框
        new_boxes = []
        unmatched_dets = boxes_image1[cent_1_indices]
        unmatched_indices = cent_1_indices

        # 确保 unmatched_dets2 和 bboxes_in_overmask 都是二维数组
        unmatched_dets2 = np.array(unmatched_dets2).reshape(-1, 6)
        # bboxes_in_overmask = np.array(bboxes_in_overmask).reshape(-1, 5)
        boxes_image1 = np.array(boxes_image1).reshape(-1, 6)

        # 步骤 1：按置信度排序，选择置信度最高的 5 个检测框
        unmatched_dets2 = unmatched_dets2[unmatched_dets2[:, 4].argsort()[::-1]]
        top5_dets2 = unmatched_dets2[:5]
        # top5_dets2 = unmatched_dets2

        # 步骤 2：将中心点映射到 image1 的坐标系
        mapped_centers = []
        for det in top5_dets2:
            x1, y1, x2, y2, score, cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0]
            mapped_centers.append((transformed_x, transformed_y))

        processed_ids = set()
        adjusted_bboxes = []
        adjusted_bboxes2 = []

        for center_x, center_y in mapped_centers:
            # 找到包含该中心点的子图
            for sub_info in sub_images_info:
                left, upper, right, lower, sub_id = sub_info['left'], sub_info['upper'], sub_info['right'], sub_info[
                    'lower'], sub_info['id']

                if left <= center_x < right and upper <= center_y < lower:
                    # 检查该子图的 id 是否已被处理
                    if sub_id in processed_ids:
                        print(f"子图像 ID {sub_id} 已被处理，跳过检测。")
                        break  # 跳过当前子图

                    # 标记该子图为已处理
                    processed_ids.add(sub_id)

                    # 子图区域对应的文件路径
                    overlap_region_path = sub_info['filename']
                    image_p = cv2.imread(overlap_region_path)

                    im0, det_bboxes_array = detector.detect(image_p)

                    iou_threshold = 0.3
                    if len(det_bboxes_array) > 0:
                        # 执行 NMS
                        indices = cv2.dnn.NMSBoxes(
                            bboxes=det_bboxes_array[:, :4].tolist(),
                            scores=det_bboxes_array[:, 4].tolist(),
                            score_threshold=0.1,  # 设置得分阈值，过滤低置信度检测
                            nms_threshold=iou_threshold
                        )
                        print("NMS indices:", indices)  # 输出 indices 以便检查

                        # 根据 NMS 结果过滤 det_bboxes_array
                        # 直接遍历 indices 中的每个索引 i
                        nms_filtered_detections = np.array([det_bboxes_array[i] for i in indices])

                    # 检查 result['det_bboxes'] 的结构
                    # det_bboxes_array = result.get('det_bboxes', [])  # 使用 get 以防键不存在
                    distance_threshold = 50
                    if len(det_bboxes_array) > 0:
                        # det_bboxes_array = det_bboxes_array[0]  # 取出第一个元素，假设是 [x_min, y_min, x_max, y_max, score]
                        edge_threshold = 10
                        # 获取子图的位置信息
                        x_offset = left
                        y_offset = upper

                        # 迭代每个 NMS 过滤后的检测框
                        for bbox in nms_filtered_detections:
                            x_min, y_min, x_max, y_max, score, lbl = bbox

                            # 计算检测框的中心点
                            # bbox_center_x = (x_min + x_max) / 2
                            # bbox_center_y = (y_min + y_max) / 2

                            # 计算中心点之间的距离
                            # distance = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)
                            adjusted_bbox = [
                                x_min + x_offset,
                                y_min + y_offset,
                                x_max + x_offset,
                                y_max + y_offset,
                                score,  # 保留置信度分数
                                lbl
                            ]
                            adjusted_bboxes2.append(adjusted_bbox)

                            # 检查是否满足距离和边缘条件
                            if (
                                    x_min >= edge_threshold and
                                    y_min >= edge_threshold and
                                    x_max <= 960 - edge_threshold and
                                    y_max <= 540 - edge_threshold):
                                # 将边界框坐标加上偏移量 (x_offset, y_offset)
                                adjusted_bbox = [
                                    x_min + x_offset,
                                    y_min + y_offset,
                                    x_max + x_offset,
                                    y_max + y_offset,
                                    score,  # 保留置信度分数
                                    lbl
                                ]
                                # adjusted_bboxes2.append(adjusted_bbox)
                                if score > thret:
                                    adjusted_bboxes.append(adjusted_bbox)

                    break  # 找到对应的子图后，跳出子图循环

        adjusted_bboxes = np.array(adjusted_bboxes)
        adjusted_bboxes2 = np.array(adjusted_bboxes2)
        # print("pppppppppppppppppppppppppppppppppp", adjusted_bboxes)
        # print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm", processed_ids)
        """
        # 将 sub_images_info 转换为按 id 索引的字典
        sub_images_dict = {sub_img['id']: sub_img for sub_img in sub_images_info}

        # 筛选符合条件的检测框
        filtered_dets = []

        for det in unmatched_dets:
            x1, y1, x2, y2, confidence, category = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            sub_id = get_sub_image_id_for_point(center_x, center_y, sub_images_dict)
            if sub_id is not None and sub_id in processed_ids:
                filtered_dets.append(det)

        # 确保 boxes_image1 是一个 NumPy 数组
        boxes_image1 = np.array(boxes_image1)
        filtered_dets = np.array(filtered_dets)
        # 初始化一个列表来存储需要添加的新检测框
        new_boxes = []
        repress_boxes = []


        for i,existing_bbox in enumerate(filtered_dets):
            # 初始化一个标志，表示是否可以添加该检测框
            can_repre = True

            # 遍历 boxes_image1 中的每个检测框
            for adjusted_bbox in adjusted_bboxes2:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                #intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                #adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                #overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.5 :
                    can_repre = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_repre:
                matches = np.where(
                    (boxes_image1[:, 0] == existing_bbox[0]) &
                    (boxes_image1[:, 1] == existing_bbox[1]) &
                    (boxes_image1[:, 2] == existing_bbox[2]) &
                    (boxes_image1[:, 3] == existing_bbox[3])
                )[0]
                boxes_image1[matches,4] *= 0.75
                repress_boxes.append(existing_bbox)
        if repress_boxes:
            repress_boxes = np.array(repress_boxes)
        """
        # 遍历 adjusted_bboxes 中的每个检测框
        for adjusted_bbox in adjusted_bboxes:
            # 初始化一个标志，表示是否可以添加该检测框
            can_add = True

            # 遍历 boxes_image1 中的每个检测框
            for existing_bbox in boxes_image1:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.2 or overlap_ratio > 0.5:
                    can_add = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_add:
                new_boxes.append(adjusted_bbox)

        # 将 new_boxes 中的检测框添加到 boxes_image1 中
        if new_boxes:
            new_boxes = np.array(new_boxes)
            boxes_image1 = np.vstack((boxes_image1, new_boxes))

        # 将 new_boxes 转换为 numpy 数组
        new_boxes = np.array(new_boxes)
        unique_dets = np.array([])
        """
        # 处理未匹配的检测框
        if len(unmatched_dets) > 8:
            change_det = np.array([])
        else:
            processed_indices = []

            # 第一次循环
            for idx, det in enumerate(unmatched_dets):
                min_index = unmatched_indices[idx]
                if det[4] < 0.5 and area[min_index] > M:
                    change_det_list.append(det)
                    boxes_image1[min_index, 4] *= k1
                    processed_indices.append(idx)

            # 第二次循环
            top_k = 5
            unprocessed_indices = [i for i in range(len(unmatched_dets)) if i not in processed_indices]
            unprocessed_dets = unmatched_dets[unprocessed_indices]
            sorted_indices = np.argsort(unprocessed_dets[:, 4])[::-1]
            processed_count = 0

            for idx in sorted_indices:
                if processed_count >= top_k:
                    break
                det = unprocessed_dets[idx]
                min_index = unmatched_indices[unprocessed_indices[idx]]
                processed_count += 1
                change_det_list.append(det)
                if area[min_index] > M:
                    boxes_image1[min_index, 4] *= k1
                else:
                    boxes_image1[min_index, 4] *= k2
                processed_indices.append(unprocessed_indices[idx])

            # 将收集的检测框转换为数组
            change_det = np.array(change_det_list) if change_det_list else np.array([])

        # 合并大框和小框的检测框
        # bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        kk = 0.5
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence
        # scores = kk * scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标


            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            #yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i],areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            # conf_thresh = np.ones_like(ovr) * 0.2
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        for i, box_keep in enumerate(keep):
            # 获取keep中当前检测框的坐标 (x1, y1, x2, y2)
            x1_k, y1_k, x2_k, y2_k = box_keep[:4]

            # 遍历det1中的每个检测框，找出坐标相同的检测框
            for box_det in dets:

                x1_d, y1_d, x2_d, y2_d, score_d, lbll = box_det

                # 如果坐标完全相等
                if x1_k == x1_d and y1_k == y1_d and x2_k == x2_d and y2_k == y2_d:
                    # 替换keep中检测框的置信度为det1中对应的置信度
                    keep[i, 4] = score_d
        # 只比较坐标部分，忽略分数
        coords_dets1 = dets[:, :4]
        coords_boxes_image1 = keep[:, :4]

        # 找到在 dets1 中但不在 boxes_image1 中的检测框（只比较坐标）
        unique_coords = []
        for coord in coords_dets1:
            if not any(np.array_equal(coord, box_coord) for box_coord in coords_boxes_image1):
                unique_coords.append(coord)

        # 根据找到的唯一坐标从 dets1 中提取完整的检测框
        unique_dets = np.array([d for d in dets if any(np.array_equal(d[:4], uc) for uc in unique_coords)])

        return keep, new_boxes, unique_dets
        """
        return boxes_image1, new_boxes, unique_dets, adjusted_bboxes2

def mve_nms2(img, image1, image2, dets, dets2, homography_matrix, iou_thresh, thre, k1, k2,detector,thret): #非重叠字图_valid_window
    boxes_image1 = dets.copy()
    boxes_image2 = dets2.copy()

    sub_images_info = split_image_single_folder(img, "/home/ubuntu/PycharmProject/Yolov7_StrongSORT_OSNet-main/image_2/" ,2,2)

    overlap_mask, overlap_region,x_offset, y_offset,area_ratio = find_overlap_region(image1, image2, homography_matrix)
    print("kllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk    area_ratio:", area_ratio)  # 输出面积比
    #bboxes_in_overmask = np.array([])

    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region_det(boxes_image1, boxes_image2,
                                                                                      homography_matrix, overlap_mask)
    print("filtered_boxes_image1", filtered_boxes_image1.size)
    print("filtered_boxes_image2", filtered_boxes_image2.size)

    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box
        change_det1 = np.array([])
        det_delete = np.array([])
        """
        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i], areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        return keep, change_det1, det_delete
        """
        return dets, change_det1, det_delete,det_delete

    else:

        M = find_big_and_small_detection_boxes_middle_det(filtered_boxes_image1)

        boxes_image1 = boxes_image1[boxes_image1[:, 4].argsort()[::-1]]
        area = (boxes_image1[:, 3] - boxes_image1[:, 1]) * (boxes_image1[:, 2] - boxes_image1[:, 0])

        cent_1x = (boxes_image1[:, 2] + boxes_image1[:, 0]) / 2
        cent_1y = (boxes_image1[:, 3] + boxes_image1[:, 1]) / 2

        cent_1 = np.stack([cent_1x, cent_1y], axis=1)
        top_k = 5
        # 初始化索引映射
        cent_1_indices = np.arange(len(cent_1))

        # 初始化变量
        unmatched_dets = []
        unmatched_dets2 = []
        matched_dets = []
        matched_dets2 = []
        change_det_list = []

        # 遍历 boxes_image2 的每个检测框
        for i, det in enumerate(boxes_image2):
            if len(cent_1) == 0:
                break

            x1, y1, x2, y2, score, cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)

            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0, 0], transformed_point[0, 0, 1]

            distances = np.sqrt((transformed_x - cent_1[:, 0]) ** 2 + (transformed_y - cent_1[:, 1]) ** 2)
            min_distance = np.min(distances)
            min_index = np.argmin(distances)

            if min_distance < thre:
                matched_index = cent_1_indices[min_index]
                matched_dets.append(boxes_image1[matched_index])
                matched_dets2.append(det)
                cent_1 = np.delete(cent_1, min_index, axis=0)
                cent_1_indices = np.delete(cent_1_indices, min_index)
            else:
                unmatched_dets2.append(det)

        # 未匹配的检测框
        new_boxes = []
        unmatched_dets = boxes_image1[cent_1_indices]
        unmatched_indices = cent_1_indices

        # 确保 unmatched_dets2 和 bboxes_in_overmask 都是二维数组
        unmatched_dets2 = np.array(unmatched_dets2).reshape(-1, 6)
        #bboxes_in_overmask = np.array(bboxes_in_overmask).reshape(-1, 5)
        boxes_image1 = np.array(boxes_image1).reshape(-1, 6)

        # 步骤 1：按置信度排序，选择置信度最高的 5 个检测框
        unmatched_dets2 = unmatched_dets2[unmatched_dets2[:, 4].argsort()[::-1]]
        #top5_dets2 = unmatched_dets2[:5]
        top5_dets2 = unmatched_dets2

        # 步骤 2：将中心点映射到 image1 的坐标系
        mapped_centers = []
        for det in top5_dets2:
            x1, y1, x2, y2, score,cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0]
            mapped_centers.append((transformed_x, transformed_y))

        processed_ids = set()
        adjusted_bboxes = []

        for center_x, center_y in mapped_centers:
            # 找到包含该中心点的子图
            for sub_info in sub_images_info:
                left, upper, right, lower, sub_id = sub_info['left'], sub_info['upper'], sub_info['right'], sub_info[
                    'lower'], sub_info['id']

                if left <= center_x < right and upper <= center_y < lower:
                    # 检查该子图的 id 是否已被处理
                    if sub_id in processed_ids:
                        print(f"子图像 ID {sub_id} 已被处理，跳过检测。")
                        break  # 跳过当前子图

                    # 标记该子图为已处理
                    processed_ids.add(sub_id)

                    # 子图区域对应的文件路径
                    overlap_region_path = sub_info['filename']
                    image_p = cv2.imread(overlap_region_path)

                    im0,det_bboxes_array = detector.detect(image_p)

                    iou_threshold = 0.3
                    if len(det_bboxes_array) > 0:
                        # 执行 NMS
                        indices = cv2.dnn.NMSBoxes(
                            bboxes=det_bboxes_array[:, :4].tolist(),
                            scores=det_bboxes_array[:, 4].tolist(),
                            score_threshold=0.1,  # 设置得分阈值，过滤低置信度检测
                            nms_threshold=iou_threshold
                        )
                        print("NMS indices:", indices)  # 输出 indices 以便检查

                        # 根据 NMS 结果过滤 det_bboxes_array
                        # 直接遍历 indices 中的每个索引 i
                        nms_filtered_detections = np.array([det_bboxes_array[i] for i in indices])

                    # 检查 result['det_bboxes'] 的结构
                    #det_bboxes_array = result.get('det_bboxes', [])  # 使用 get 以防键不存在
                    distance_threshold = 50
                    if len(det_bboxes_array) > 0:
                        #det_bboxes_array = det_bboxes_array[0]  # 取出第一个元素，假设是 [x_min, y_min, x_max, y_max, score]
                        edge_threshold = 10
                        # 获取子图的位置信息
                        x_offset = left
                        y_offset = upper

                        # 迭代每个 NMS 过滤后的检测框
                        for bbox in nms_filtered_detections:
                            x_min, y_min, x_max, y_max, score, lbl = bbox

                            # 计算检测框的中心点
                            #bbox_center_x = (x_min + x_max) / 2
                            #bbox_center_y = (y_min + y_max) / 2

                            # 计算中心点之间的距离
                            #distance = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)

                            # 检查是否满足距离和边缘条件
                            if (
                                    x_min >= edge_threshold and
                                    y_min >= edge_threshold and
                                    x_max <= 960 - edge_threshold and
                                    y_max <= 540 - edge_threshold):
                                # 将边界框坐标加上偏移量 (x_offset, y_offset)
                                adjusted_bbox = [
                                    x_min + x_offset,
                                    y_min + y_offset,
                                    x_max + x_offset,
                                    y_max + y_offset,
                                    score,  # 保留置信度分数
                                    lbl
                                ]
                                if score > thret:
                                    adjusted_bboxes.append(adjusted_bbox)

                    break  # 找到对应的子图后，跳出子图循环

        adjusted_bboxes = np.array(adjusted_bboxes)
        print("pppppppppppppppppppppppppppppppppp", adjusted_bboxes)

        # 确保 boxes_image1 是一个 NumPy 数组
        boxes_image1 = np.array(boxes_image1)

        # 初始化一个列表来存储需要添加的新检测框
        new_boxes = []

        # 遍历 adjusted_bboxes 中的每个检测框
        for adjusted_bbox in adjusted_bboxes:
            # 初始化一个标志，表示是否可以添加该检测框
            can_add = True

            # 遍历 boxes_image1 中的每个检测框
            for i, existing_bbox in enumerate(boxes_image1):
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.2 or overlap_ratio > 0.5:
                    can_add = False

                    break  # 跳出循环，因为不需要继续检查

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_add:
                new_boxes.append(adjusted_bbox)

        # 将 new_boxes 中的检测框添加到 boxes_image1 中
        if new_boxes:
            new_boxes = np.array(new_boxes)
            boxes_image1 = np.vstack((boxes_image1, new_boxes))

        #将 new_boxes 转换为 numpy 数组
        new_boxes = np.array(new_boxes)
        unique_dets = np.array([])
        """
        # 处理未匹配的检测框
        if len(unmatched_dets) > 8:
            change_det = np.array([])
        else:
            processed_indices = []

            # 第一次循环
            for idx, det in enumerate(unmatched_dets):
                min_index = unmatched_indices[idx]
                if det[4] < 0.5 and area[min_index] > M:
                    change_det_list.append(det)
                    boxes_image1[min_index, 4] *= k1
                    processed_indices.append(idx)

            # 第二次循环
            top_k = 5
            unprocessed_indices = [i for i in range(len(unmatched_dets)) if i not in processed_indices]
            unprocessed_dets = unmatched_dets[unprocessed_indices]
            sorted_indices = np.argsort(unprocessed_dets[:, 4])[::-1]
            processed_count = 0

            for idx in sorted_indices:
                if processed_count >= top_k:
                    break
                det = unprocessed_dets[idx]
                min_index = unmatched_indices[unprocessed_indices[idx]]
                processed_count += 1
                change_det_list.append(det)
                if area[min_index] > M:
                    boxes_image1[min_index, 4] *= k1
                else:
                    boxes_image1[min_index, 4] *= k2
                processed_indices.append(unprocessed_indices[idx])

            # 将收集的检测框转换为数组
            change_det = np.array(change_det_list) if change_det_list else np.array([])

        # 合并大框和小框的检测框
        # bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        kk = 0.5
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence
        # scores = kk * scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标


            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            #yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i],areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            # conf_thresh = np.ones_like(ovr) * 0.2
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        for i, box_keep in enumerate(keep):
            # 获取keep中当前检测框的坐标 (x1, y1, x2, y2)
            x1_k, y1_k, x2_k, y2_k = box_keep[:4]

            # 遍历det1中的每个检测框，找出坐标相同的检测框
            for box_det in dets:

                x1_d, y1_d, x2_d, y2_d, score_d, lbll = box_det

                # 如果坐标完全相等
                if x1_k == x1_d and y1_k == y1_d and x2_k == x2_d and y2_k == y2_d:
                    # 替换keep中检测框的置信度为det1中对应的置信度
                    keep[i, 4] = score_d
        # 只比较坐标部分，忽略分数
        coords_dets1 = dets[:, :4]
        coords_boxes_image1 = keep[:, :4]

        # 找到在 dets1 中但不在 boxes_image1 中的检测框（只比较坐标）
        unique_coords = []
        for coord in coords_dets1:
            if not any(np.array_equal(coord, box_coord) for box_coord in coords_boxes_image1):
                unique_coords.append(coord)

        # 根据找到的唯一坐标从 dets1 中提取完整的检测框
        unique_dets = np.array([d for d in dets if any(np.array_equal(d[:4], uc) for uc in unique_coords)])

        return keep, new_boxes, unique_dets
        """
        return boxes_image1, new_boxes, unique_dets,adjusted_bboxes



def mve_nms3(img, image1, image2, dets, dets2, homography_matrix, iou_thresh, thre, k1, k2, detector, th):  #free_window
    boxes_image1 = dets.copy()
    boxes_image2 = dets2.copy()

    #sub_images_info = split_image_single_folder(img,
    #                                            "/home/ubuntu/PycharmProject/Yolov7_StrongSORT_OSNet-main/image_1/", 2,
    #                                            2)

    overlap_mask, overlap_region, x_offset, y_offset, area_ratio = find_overlap_region(image1, image2,
                                                                                       homography_matrix)
    print("kllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk    area_ratio:", area_ratio)  # 输出面积比
    # bboxes_in_overmask = np.array([])

    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region_det_cen(boxes_image1, boxes_image2,
                                                                                      homography_matrix, overlap_mask)
    print("filtered_boxes_image1", filtered_boxes_image1.size)
    print("filtered_boxes_image2", filtered_boxes_image2.size)

    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box
        change_det1 = np.array([])
        det_delete = np.array([])
        """
        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i], areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        return keep, change_det1, det_delete
        """
        return dets, change_det1, det_delete, keep

    else:

        M = find_big_and_small_detection_boxes_middle_det(filtered_boxes_image1)

        filtered_boxes_image1 = filtered_boxes_image1[filtered_boxes_image1[:, 4].argsort()[::-1]]
        area = (filtered_boxes_image1[:, 3] - filtered_boxes_image1[:, 1]) * (filtered_boxes_image1[:, 2] - filtered_boxes_image1[:, 0])

        cent_1x = (filtered_boxes_image1[:, 2] + filtered_boxes_image1[:, 0]) / 2
        cent_1y = (filtered_boxes_image1[:, 3] + filtered_boxes_image1[:, 1]) / 2

        cent_1 = np.stack([cent_1x, cent_1y], axis=1)
        top_k = 5
        # 初始化索引映射
        cent_1_indices = np.arange(len(cent_1))

        # 初始化变量
        unmatched_dets = []
        unmatched_dets2 = []
        matched_dets = []
        matched_dets2 = []
        change_det_list = []

        # 遍历 boxes_image2 的每个检测框
        for i, det in enumerate(filtered_boxes_image2):
            if len(cent_1) == 0:
                break

            x1, y1, x2, y2, score, cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)

            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0, 0], transformed_point[0, 0, 1]

            distances = np.sqrt((transformed_x - cent_1[:, 0]) ** 2 + (transformed_y - cent_1[:, 1]) ** 2)
            min_distance = np.min(distances)
            min_index = np.argmin(distances)

            if min_distance < thre:
                matched_index = cent_1_indices[min_index]
                matched_dets.append(filtered_boxes_image1[matched_index])
                matched_dets2.append(det)
                cent_1 = np.delete(cent_1, min_index, axis=0)
                cent_1_indices = np.delete(cent_1_indices, min_index)
            else:
                unmatched_dets2.append(det)

        # 未匹配的检测框
        new_boxes = []
        unmatched_dets = boxes_image1[cent_1_indices]
        unmatched_indices = cent_1_indices

        # 确保 unmatched_dets2 和 bboxes_in_overmask 都是二维数组
        unmatched_dets2 = np.array(unmatched_dets2).reshape(-1, 6)
        # bboxes_in_overmask = np.array(bboxes_in_overmask).reshape(-1, 5)
        boxes_image1 = np.array(boxes_image1).reshape(-1, 6)

        # 步骤 1：按置信度排序，选择置信度最高的 5 个检测框
        unmatched_dets2 = unmatched_dets2[unmatched_dets2[:, 4].argsort()[::-1]]
        top = min(10,len(unmatched_dets2))
        top5_dets2 = unmatched_dets2[:top]
        #top5_dets2 = unmatched_dets2

        # 步骤 2：将中心点映射到 image1 的坐标系
        mapped_centers = []
        for det in top5_dets2:
            x1, y1, x2, y2, score, cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0]
            mapped_centers.append((transformed_x, transformed_y))

        processed_ids = set()
        adjusted_bboxes = []
        adjusted_bboxes2 = []

        for center_x, center_y in mapped_centers:
            if not (0 <= center_x <= 1920 and 0 <= center_y <= 1080):
                print(f"Center coordinates ({center_x}, {center_y}) out of bounds.")
                continue  # 跳过此循环

            crop_width = 960
            crop_height = 540

            x_start = int(center_x - crop_width / 2)
            y_start = int(center_y - crop_height / 2)
            x_end = int(center_x + crop_width / 2)
            y_end = int(center_y + crop_height / 2)

            # 图像的原始大小
            height, width = image1.shape[:2]

            # 初始化全黑背景图像
            cropped_image = np.zeros((crop_height, crop_width, 3), dtype=image1.dtype)

            # 计算有效的裁剪范围
            x_start_img = max(0, x_start)
            y_start_img = max(0, y_start)
            x_end_img = min(width, x_end)
            y_end_img = min(height, y_end)

            # 计算在黑色背景上的填充区域
            x_start_cropped = max(0, -x_start)
            y_start_cropped = max(0, -y_start)
            x_end_cropped = x_start_cropped + (x_end_img - x_start_img)
            y_end_cropped = y_start_cropped + (y_end_img - y_start_img)

            # 填充有效区域的像素值
            cropped_image[
            y_start_cropped:y_end_cropped, x_start_cropped:x_end_cropped
            ] = image1[y_start_img:y_end_img, x_start_img:x_end_img]

            if cropped_image.size == 0:
                print("Empty cropped image at coordinates:", x_start, y_start, x_end, y_end)
                continue

            # 调用目标检测模型
            im0, det_bboxes_array = detector.detect(cropped_image)

            iou_threshold = 0.3
            if len(det_bboxes_array) > 0:

                # 按置信度分数降序排序
                #high_conf_detections = high_conf_detections[high_conf_detections[:, 4].argsort()[::-1]]

                # 如果少于5个，则全部保留，否则只保留前五个
                #if len(high_conf_detections) > 5:
                #    nms_filtered_detections = high_conf_detections[:5]
                #else:
                #    nms_filtered_detections = high_conf_detections

                edge_threshold = 10
                for bbox in det_bboxes_array:
                    x_min, y_min, x_max, y_max, score, lbl = bbox
                    x_offset = x_start
                    y_offset = y_start
                    # 计算检测框的中心点
                    # bbox_center_x = (x_min + x_max) / 2
                    # bbox_center_y = (y_min + y_max) / 2

                    # 计算中心点之间的距离
                    # distance = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)
                    # 检查是否满足距离和边缘条件
                    if (
                            x_min >= edge_threshold and
                            y_min >= edge_threshold and
                            x_max <= crop_width - edge_threshold and
                            y_max <= crop_height - edge_threshold):
                        # 将边界框坐标加上偏移量 (x_offset, y_offset)
                        adjusted_bbox = [
                            x_min + x_offset,
                            y_min + y_offset,
                            x_max + x_offset,
                            y_max + y_offset,
                            score,  # 保留置信度分数
                            lbl
                        ]
                        # adjusted_bboxes2.append(adjusted_bbox)
                        if score > th:
                            adjusted_bboxes.append(adjusted_bbox)
        adjusted_bboxes = np.array(adjusted_bboxes)
        #print("llllllllllllllllllllllllll", adjusted_bboxes)
        #adjusted_bboxes2 = np.array(adjusted_bboxes)
        keep_indices = []
        num_boxes = len(adjusted_bboxes)
        """
        for i in range(num_boxes):
            box_i = adjusted_bboxes[i]
            has_large_iou = False
            for j in range(num_boxes):
                if i == j:
                    continue  # 跳过自身
                box_j = adjusted_bboxes[j]
                iou = compute_iou(box_i, box_j)
                if iou > iou_threshold:
                    has_large_iou = True
                    break  # 找到一个满足条件的框即可
            if has_large_iou:
                keep_indices.append(i)
        
        # 保留满足条件的检测框
        adjusted_bboxes = adjusted_bboxes[keep_indices]
        print("llllllllllllllllllllllllll", adjusted_bboxes)
        """
        iou_threshold = 0.3
        if len(adjusted_bboxes) > 0:
            # 执行 NMS
            indices = cv2.dnn.NMSBoxes(
                bboxes=adjusted_bboxes[:, :4].tolist(),
                scores=adjusted_bboxes[:, 4].tolist(),
                score_threshold=0.1,  # 设置得分阈值，过滤低置信度检测
                nms_threshold=iou_threshold
            )
            print("NMS indices:", indices)  # 输出 indices 以便检查

            # 根据 NMS 结果过滤 det_bboxes_array
            # 直接遍历 indices 中的每个索引 i
            adjusted_bboxes = np.array([adjusted_bboxes[i] for i in indices])

        #print("llllllllllllllllllllllllll", adjusted_bboxes)
        adjusted_bboxes2 = np.array(adjusted_bboxes)


        # print("pppppppppppppppppppppppppppppppppp", adjusted_bboxes)
        # print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm", processed_ids)
        """
        # 将 sub_images_info 转换为按 id 索引的字典
        sub_images_dict = {sub_img['id']: sub_img for sub_img in sub_images_info}

        # 筛选符合条件的检测框
        filtered_dets = []

        for det in unmatched_dets:
            x1, y1, x2, y2, confidence, category = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            sub_id = get_sub_image_id_for_point(center_x, center_y, sub_images_dict)
            if sub_id is not None and sub_id in processed_ids:
                filtered_dets.append(det)

        # 确保 boxes_image1 是一个 NumPy 数组
        boxes_image1 = np.array(boxes_image1)
        filtered_dets = np.array(filtered_dets)
        # 初始化一个列表来存储需要添加的新检测框
        new_boxes = []
        repress_boxes = []


        for i,existing_bbox in enumerate(filtered_dets):
            # 初始化一个标志，表示是否可以添加该检测框
            can_repre = True

            # 遍历 boxes_image1 中的每个检测框
            for adjusted_bbox in adjusted_bboxes2:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                #intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                #adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                #overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.5 :
                    can_repre = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_repre:
                matches = np.where(
                    (boxes_image1[:, 0] == existing_bbox[0]) &
                    (boxes_image1[:, 1] == existing_bbox[1]) &
                    (boxes_image1[:, 2] == existing_bbox[2]) &
                    (boxes_image1[:, 3] == existing_bbox[3])
                )[0]
                boxes_image1[matches,4] *= 0.75
                repress_boxes.append(existing_bbox)
        if repress_boxes:
            repress_boxes = np.array(repress_boxes)
        """
        # 遍历 adjusted_bboxes 中的每个检测框
        for adjusted_bbox in adjusted_bboxes:
            # 初始化一个标志，表示是否可以添加该检测框
            can_add = True

            # 遍历 boxes_image1 中的每个检测框
            for existing_bbox in boxes_image1:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.2 or overlap_ratio > 0.5:
                    can_add = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_add:
                new_boxes.append(adjusted_bbox)

        # 将 new_boxes 中的检测框添加到 boxes_image1 中
        if new_boxes:
            new_boxes = np.array(new_boxes)
            boxes_image1 = np.vstack((boxes_image1, new_boxes))

        # 将 new_boxes 转换为 numpy 数组
        new_boxes = np.array(new_boxes)
        print("kkkkkkkkkkkkkkkkkkkkkkkkk",adjusted_bboxes)
        #print("oooooooooooooooooooooooooooooo", new_boxes)
        unique_dets = np.array([])
        """
        # 处理未匹配的检测框
        if len(unmatched_dets) > 8:
            change_det = np.array([])
        else:
            processed_indices = []

            # 第一次循环
            for idx, det in enumerate(unmatched_dets):
                min_index = unmatched_indices[idx]
                if det[4] < 0.5 and area[min_index] > M:
                    change_det_list.append(det)
                    boxes_image1[min_index, 4] *= k1
                    processed_indices.append(idx)

            # 第二次循环
            top_k = 5
            unprocessed_indices = [i for i in range(len(unmatched_dets)) if i not in processed_indices]
            unprocessed_dets = unmatched_dets[unprocessed_indices]
            sorted_indices = np.argsort(unprocessed_dets[:, 4])[::-1]
            processed_count = 0

            for idx in sorted_indices:
                if processed_count >= top_k:
                    break
                det = unprocessed_dets[idx]
                min_index = unmatched_indices[unprocessed_indices[idx]]
                processed_count += 1
                change_det_list.append(det)
                if area[min_index] > M:
                    boxes_image1[min_index, 4] *= k1
                else:
                    boxes_image1[min_index, 4] *= k2
                processed_indices.append(unprocessed_indices[idx])

            # 将收集的检测框转换为数组
            change_det = np.array(change_det_list) if change_det_list else np.array([])

        # 合并大框和小框的检测框
        # bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        kk = 0.5
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence
        # scores = kk * scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标


            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            #yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i],areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            # conf_thresh = np.ones_like(ovr) * 0.2
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        for i, box_keep in enumerate(keep):
            # 获取keep中当前检测框的坐标 (x1, y1, x2, y2)
            x1_k, y1_k, x2_k, y2_k = box_keep[:4]

            # 遍历det1中的每个检测框，找出坐标相同的检测框
            for box_det in dets:

                x1_d, y1_d, x2_d, y2_d, score_d, lbll = box_det

                # 如果坐标完全相等
                if x1_k == x1_d and y1_k == y1_d and x2_k == x2_d and y2_k == y2_d:
                    # 替换keep中检测框的置信度为det1中对应的置信度
                    keep[i, 4] = score_d
        # 只比较坐标部分，忽略分数
        coords_dets1 = dets[:, :4]
        coords_boxes_image1 = keep[:, :4]

        # 找到在 dets1 中但不在 boxes_image1 中的检测框（只比较坐标）
        unique_coords = []
        for coord in coords_dets1:
            if not any(np.array_equal(coord, box_coord) for box_coord in coords_boxes_image1):
                unique_coords.append(coord)

        # 根据找到的唯一坐标从 dets1 中提取完整的检测框
        unique_dets = np.array([d for d in dets if any(np.array_equal(d[:4], uc) for uc in unique_coords)])

        return keep, new_boxes, unique_dets
        """
        return boxes_image1, new_boxes, unique_dets, adjusted_bboxes


def mve_nms4(img, image1, image2, dets, dets2, homography_matrix, iou_thresh, thre, k1, k2, detector,th): #valid_window
    boxes_image1 = dets.copy()
    boxes_image2 = dets2.copy()

    sub_images_info = split_image_single_folder_overlap(img,
                                                "/home/ubuntu/PycharmProject/Yolov7_StrongSORT_OSNet-main/image_1/", 2,
                                                2)

    overlap_mask, overlap_region, x_offset, y_offset, area_ratio = find_overlap_region(image1, image2,
                                                                                       homography_matrix)
    print("kllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk    area_ratio:", area_ratio)  # 输出面积比
    # bboxes_in_overmask = np.array([])

    # centers_image1 = compute_center_of_boxes(boxes_image1)
    # centers_image2 = compute_center_of_boxes(boxes_image2)

    # boxes_image1_1 = find_small_detection_boxes(boxes_image1, 2.5)
    # centers_image1 = compute_center_of_boxes(boxes_image1_1)
    # centers_image1 = compute_center_of_boxes(boxes_image1)

    # filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region(boxes_image1, centers_image1,
    # boxes_image2, centers_image2,
    # homography_matrix, overlap_mask)
    filtered_boxes_image1, filtered_boxes_image2 = filter_boxes_in_overlap_region_det(boxes_image1, boxes_image2,
                                                                                      homography_matrix, overlap_mask)
    print("filtered_boxes_image1", filtered_boxes_image1.size)
    print("filtered_boxes_image2", filtered_boxes_image2.size)

    if filtered_boxes_image1.shape[0] == 0 or filtered_boxes_image2.shape[0] == 0:
        print("No overlap detected or no filtered boxes found.")
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box
        change_det1 = np.array([])
        det_delete = np.array([])
        """
        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i], areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        return keep, change_det1, det_delete
        """
        return dets, change_det1, keep, det_delete

    else:

        M = find_big_and_small_detection_boxes_middle_det(filtered_boxes_image1)

        boxes_image1 = boxes_image1[boxes_image1[:, 4].argsort()[::-1]]
        area = (boxes_image1[:, 3] - boxes_image1[:, 1]) * (boxes_image1[:, 2] - boxes_image1[:, 0])

        cent_1x = (boxes_image1[:, 2] + boxes_image1[:, 0]) / 2
        cent_1y = (boxes_image1[:, 3] + boxes_image1[:, 1]) / 2

        cent_1 = np.stack([cent_1x, cent_1y], axis=1)
        top_k = 5
        # 初始化索引映射
        cent_1_indices = np.arange(len(cent_1))

        # 初始化变量
        unmatched_dets = []
        unmatched_dets2 = []
        matched_dets = []
        matched_dets2 = []
        change_det_list = []

        # 遍历 boxes_image2 的每个检测框
        for i, det in enumerate(boxes_image2):
            if len(cent_1) == 0:
                break

            x1, y1, x2, y2, score, cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)

            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0, 0], transformed_point[0, 0, 1]

            distances = np.sqrt((transformed_x - cent_1[:, 0]) ** 2 + (transformed_y - cent_1[:, 1]) ** 2)
            min_distance = np.min(distances)
            min_index = np.argmin(distances)

            if min_distance < thre:
                matched_index = cent_1_indices[min_index]
                matched_dets.append(boxes_image1[matched_index])
                matched_dets2.append(det)
                cent_1 = np.delete(cent_1, min_index, axis=0)
                cent_1_indices = np.delete(cent_1_indices, min_index)
            else:
                unmatched_dets2.append(det)

        # 未匹配的检测框
        new_boxes = []
        unmatched_dets = boxes_image1[cent_1_indices]
        unmatched_indices = cent_1_indices

        # 确保 unmatched_dets2 和 bboxes_in_overmask 都是二维数组
        unmatched_dets2 = np.array(unmatched_dets2).reshape(-1, 6)
        # bboxes_in_overmask = np.array(bboxes_in_overmask).reshape(-1, 5)
        boxes_image1 = np.array(boxes_image1).reshape(-1, 6)

        # 步骤 1：按置信度排序，选择置信度最高的 5 个检测框
        unmatched_dets2 = unmatched_dets2[unmatched_dets2[:, 4].argsort()[::-1]]
        # top5_dets2 = unmatched_dets2[:5]
        top5_dets2 = unmatched_dets2

        # 步骤 2：将中心点映射到 image1 的坐标系
        mapped_centers = []
        for det in top5_dets2:
            x1, y1, x2, y2, score, cata = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(center_point, homography_matrix)
            transformed_x, transformed_y = transformed_point[0, 0]
            mapped_centers.append((transformed_x, transformed_y))

        processed_ids = set()
        adjusted_bboxes = []
        adjusted_bboxes2 = []

        for center_x, center_y in mapped_centers:
            # 检查 center_x 和 center_y 是否在有效范围内
            if center_x < 0 or center_x > 1920 or center_y < 0 or center_y > 1080:
                print(f"中心点 ({center_x}, {center_y}) 超出图像范围，跳过此点。")
                continue  # 跳过该点

            # 初始化最小距离和最近的子图信息
            min_distance = float('inf')
            nearest_sub_info = None

            # 遍历所有子图，找到与当前中心点最近的子图
            for sub_info in sub_images_info:
                left = sub_info['left']
                upper = sub_info['upper']
                right = sub_info['right']
                lower = sub_info['lower']
                sub_id = sub_info['id']

                # 计算子图的中心点坐标
                sub_center_x = (left + right) / 2
                sub_center_y = (upper + lower) / 2

                # 计算当前中心点与子图中心点的欧氏距离
                distance = ((center_x - sub_center_x) ** 2 + (center_y - sub_center_y) ** 2) ** 0.5

                # 更新最近的子图
                if distance < min_distance:
                    min_distance = distance
                    nearest_sub_info = sub_info

            if nearest_sub_info is not None:
                left = nearest_sub_info['left']
                upper = nearest_sub_info['upper']
                right = nearest_sub_info['right']
                lower = nearest_sub_info['lower']
                sub_id = nearest_sub_info['id']
                # 检查该子图的 id 是否已被处理
                if sub_id in processed_ids:
                    print(f"子图像 ID {sub_id} 已被处理，跳过检测。")
                    continue  # 跳过当前中心点

                # 标记该子图为已处理
                processed_ids.add(sub_id)
                # 子图区域对应的文件路径
                overlap_region_path = nearest_sub_info['filename']
                image_p = cv2.imread(overlap_region_path)

                im0, det_bboxes_array = detector.detect(image_p)

                iou_threshold = 0.3
                if len(det_bboxes_array) > 0:
                        # 执行 NMS
                    indices = cv2.dnn.NMSBoxes(
                            bboxes=det_bboxes_array[:, :4].tolist(),
                            scores=det_bboxes_array[:, 4].tolist(),
                            score_threshold=0.1,  # 设置得分阈值，过滤低置信度检测
                            nms_threshold=iou_threshold
                        )
                    print("NMS indices:", indices)  # 输出 indices 以便检查

                        # 根据 NMS 结果过滤 det_bboxes_array
                        # 直接遍历 indices 中的每个索引 i
                    nms_filtered_detections = np.array([det_bboxes_array[i] for i in indices])

                    # 检查 result['det_bboxes'] 的结构
                    # det_bboxes_array = result.get('det_bboxes', [])  # 使用 get 以防键不存在
                    distance_threshold = 50
                if len(det_bboxes_array) > 0:
                        # det_bboxes_array = det_bboxes_array[0]  # 取出第一个元素，假设是 [x_min, y_min, x_max, y_max, score]
                    edge_threshold = 10
                        # 获取子图的位置信息
                    x_offset = left
                    y_offset = upper

                        # 迭代每个 NMS 过滤后的检测框
                    for bbox in nms_filtered_detections:
                        x_min, y_min, x_max, y_max, score, lbl = bbox

                            # 计算检测框的中心点
                            # bbox_center_x = (x_min + x_max) / 2
                            # bbox_center_y = (y_min + y_max) / 2

                            # 计算中心点之间的距离
                            # distance = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)
                        adjusted_bbox = [
                                x_min + x_offset,
                                y_min + y_offset,
                                x_max + x_offset,
                                y_max + y_offset,
                                score,  # 保留置信度分数
                                lbl
                            ]
                        adjusted_bboxes2.append(adjusted_bbox)

                            # 检查是否满足距离和边缘条件
                        if (
                                x_min >= edge_threshold and
                                y_min >= edge_threshold and
                                x_max <= 960 - edge_threshold and
                                y_max <= 540 - edge_threshold):
                                # 将边界框坐标加上偏移量 (x_offset, y_offset)
                            adjusted_bbox = [
                                    x_min + x_offset,
                                    y_min + y_offset,
                                    x_max + x_offset,
                                    y_max + y_offset,
                                    score,  # 保留置信度分数
                                    lbl
                                ]
                            # adjusted_bboxes2.append(adjusted_bbox)
                            if score > th:
                                adjusted_bboxes.append(adjusted_bbox)

                break  # 找到对应的子图后，跳出子图循环

        adjusted_bboxes = np.array(adjusted_bboxes)
        adjusted_bboxes2 = np.array(adjusted_bboxes2)
        # print("pppppppppppppppppppppppppppppppppp", adjusted_bboxes)
        # print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm", processed_ids)
        """
        # 将 sub_images_info 转换为按 id 索引的字典
        sub_images_dict = {sub_img['id']: sub_img for sub_img in sub_images_info}

        # 筛选符合条件的检测框
        filtered_dets = []

        for det in unmatched_dets:
            x1, y1, x2, y2, confidence, category = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            sub_id = get_sub_image_id_for_point(center_x, center_y, sub_images_dict)
            if sub_id is not None and sub_id in processed_ids:
                filtered_dets.append(det)

        # 确保 boxes_image1 是一个 NumPy 数组
        boxes_image1 = np.array(boxes_image1)
        filtered_dets = np.array(filtered_dets)
        # 初始化一个列表来存储需要添加的新检测框
        new_boxes = []
        repress_boxes = []


        for i,existing_bbox in enumerate(filtered_dets):
            # 初始化一个标志，表示是否可以添加该检测框
            can_repre = True

            # 遍历 boxes_image1 中的每个检测框
            for adjusted_bbox in adjusted_bboxes2:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                #intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                #adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                #overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.5 :
                    can_repre = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_repre:
                matches = np.where(
                    (boxes_image1[:, 0] == existing_bbox[0]) &
                    (boxes_image1[:, 1] == existing_bbox[1]) &
                    (boxes_image1[:, 2] == existing_bbox[2]) &
                    (boxes_image1[:, 3] == existing_bbox[3])
                )[0]
                boxes_image1[matches,4] *= 0.75
                repress_boxes.append(existing_bbox)
        if repress_boxes:
            repress_boxes = np.array(repress_boxes)
        """
        # 遍历 adjusted_bboxes 中的每个检测框
        for adjusted_bbox in adjusted_bboxes:
            # 初始化一个标志，表示是否可以添加该检测框
            can_add = True

            # 遍历 boxes_image1 中的每个检测框
            for existing_bbox in boxes_image1:
                # 计算 IoU
                iou = calculate_iou(adjusted_bbox, existing_bbox)
                intersection_area = calculate_intersection(adjusted_bbox, existing_bbox)

                # 计算与 adjusted_bbox 的重叠比
                adjusted_bbox_area = (adjusted_bbox[2] - adjusted_bbox[0]) * (adjusted_bbox[3] - adjusted_bbox[1])
                overlap_ratio = intersection_area / adjusted_bbox_area if adjusted_bbox_area > 0 else 0

                # 如果 IoU >= 0.2 或重叠比 > 0.5，则不添加该检测框
                if iou >= 0.2 or overlap_ratio > 0.5:
                    can_add = False
                    break  # 不需要继续检查，跳出循环

            # 如果可以添加，将该检测框添加到 new_boxes 列表中
            if can_add:
                new_boxes.append(adjusted_bbox)

        # 将 new_boxes 中的检测框添加到 boxes_image1 中
        if new_boxes:
            new_boxes = np.array(new_boxes)
            boxes_image1 = np.vstack((boxes_image1, new_boxes))

        # 将 new_boxes 转换为 numpy 数组
        new_boxes = np.array(new_boxes)
        unique_dets = np.array([])
        """
        # 处理未匹配的检测框
        if len(unmatched_dets) > 8:
            change_det = np.array([])
        else:
            processed_indices = []

            # 第一次循环
            for idx, det in enumerate(unmatched_dets):
                min_index = unmatched_indices[idx]
                if det[4] < 0.5 and area[min_index] > M:
                    change_det_list.append(det)
                    boxes_image1[min_index, 4] *= k1
                    processed_indices.append(idx)

            # 第二次循环
            top_k = 5
            unprocessed_indices = [i for i in range(len(unmatched_dets)) if i not in processed_indices]
            unprocessed_dets = unmatched_dets[unprocessed_indices]
            sorted_indices = np.argsort(unprocessed_dets[:, 4])[::-1]
            processed_count = 0

            for idx in sorted_indices:
                if processed_count >= top_k:
                    break
                det = unprocessed_dets[idx]
                min_index = unmatched_indices[unprocessed_indices[idx]]
                processed_count += 1
                change_det_list.append(det)
                if area[min_index] > M:
                    boxes_image1[min_index, 4] *= k1
                else:
                    boxes_image1[min_index, 4] *= k2
                processed_indices.append(unprocessed_indices[idx])

            # 将收集的检测框转换为数组
            change_det = np.array(change_det_list) if change_det_list else np.array([])

        # 合并大框和小框的检测框
        # bboxes_image_f = np.concatenate((boxes_image1[matching_indices1], boxes_image1[matching_indices2]), axis=0)

        kk = 0.5
        x1 = boxes_image1[:, 0]  # xmin
        y1 = boxes_image1[:, 1]  # ymin
        x2 = boxes_image1[:, 2]  # xmax
        y2 = boxes_image1[:, 3]  # ymax
        scores = boxes_image1[:, 4]  # confidence
        # scores = kk * scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
        order = scores.argsort()[::-1]  # bounding box的置信度排序
        keep = np.array([])  # 用来保存最后留下来的bounding box

        while order.size > 0:
            i = order[0]  # 置信度最高的bounding box的index
            if keep.size == 0:
                keep = np.array([boxes_image1[i]])
            else:
                keep = np.append(keep, [boxes_image1[i]], axis=0)  # 添加本次置信度最高的bounding box的index

            # 当前bbox和剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标


            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            #yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # sou = inter / np.minimum(areas[i],areas[order[1:]])
            # ovr = ovr * 0.5 + sou * 0.5

            # 只对 IoU 大于阈值的框进行加权
            # inds_iou_above_thresh = np.where(ovr > iou_thresh)[0]
            # scores[order[inds_iou_above_thresh + 1]] = scores[order[inds_iou_above_thresh + 1]] - (1 - kk) * ovr[
            # inds_iou_above_thresh]

            # 根据 IoU 计算置信度阈值
            conf_thresh = np.zeros_like(ovr)  # 初始化为0
            # conf_thresh = ovr
            # conf_thresh = np.ones_like(ovr) * 0.2
            conf_thresh[ovr > iou_thresh] = ovr[ovr > iou_thresh]

            # 找到满足以下条件的index:
            # 1. IoU 小于或等于 iou_thresh
            # 2. 或者置信度大于等于计算得到的 conf_thresh
            inds = np.where(scores[order[1:]] >= conf_thresh)[0]

            # 更新order数组
            order = order[inds + 1]

        for i, box_keep in enumerate(keep):
            # 获取keep中当前检测框的坐标 (x1, y1, x2, y2)
            x1_k, y1_k, x2_k, y2_k = box_keep[:4]

            # 遍历det1中的每个检测框，找出坐标相同的检测框
            for box_det in dets:

                x1_d, y1_d, x2_d, y2_d, score_d, lbll = box_det

                # 如果坐标完全相等
                if x1_k == x1_d and y1_k == y1_d and x2_k == x2_d and y2_k == y2_d:
                    # 替换keep中检测框的置信度为det1中对应的置信度
                    keep[i, 4] = score_d
        # 只比较坐标部分，忽略分数
        coords_dets1 = dets[:, :4]
        coords_boxes_image1 = keep[:, :4]

        # 找到在 dets1 中但不在 boxes_image1 中的检测框（只比较坐标）
        unique_coords = []
        for coord in coords_dets1:
            if not any(np.array_equal(coord, box_coord) for box_coord in coords_boxes_image1):
                unique_coords.append(coord)

        # 根据找到的唯一坐标从 dets1 中提取完整的检测框
        unique_dets = np.array([d for d in dets if any(np.array_equal(d[:4], uc) for uc in unique_coords)])

        return keep, new_boxes, unique_dets
        """
        return boxes_image1, new_boxes, unique_dets, adjusted_bboxes2


def all_nms_plus(dets, iou_thresh):
    x1 = dets[:, 1]  # xmin
    y1 = dets[:, 2]  # ymin
    x2 = dets[:, 3]  # xmax
    y2 = dets[:, 4]  # ymax
    scores = dets[:, 5]  # confidence

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个bounding box的面积
    order = scores.argsort()[::-1]  # bounding box的置信度排序
    keep = np.array([])  # 用来保存最后留下来的bounding box

    while order.size > 0:
        i = order[0]  # 置信度最高的bounding box的index
        if keep.size == 0:
            keep = np.array([dets[i]])
        else:
            keep = np.append(keep, [dets[i]], axis=0)  # 添加本次置信度最高的bounding box的index

        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 根据IoU计算置信度阈值
        conf_thresh = (1 + ovr) / 2

        # 找到满足以下条件的index:
        # 1. IoU小于或等于iou_thresh
        # 2. 或者置信度大于等于计算得到的conf_thresh
        inds = np.where((ovr <= iou_thresh) | (scores[order[1:]] >= conf_thresh))[0]

        # 更新order数组
        order = order[inds + 1]

    return keep

def all_nms(dets, thresh):
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax
    scores = dets[:, 4]  # confidence

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个boundingbox的面积
    order = scores.argsort()[::-1]  # boundingbox的置信度排序
    keep = np.array([])  # 用来保存最后留下来的boundingbox
    while order.size > 0:
        i = order[0]  # 置信度最高的boundingbox的index
        if keep.size == 0:
            keep = np.array([dets[i]])
        else:
            keep = np.append(keep, [dets[i]], axis=0)  # # 添加本次置信度最高的boundingbox的index

        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def soft_nms_a(dets, thresh, sigma=0.5, method=1):
    # 提取坐标和分数
    x1 = dets[:, 1]  # xmin
    y1 = dets[:, 2]  # ymin
    x2 = dets[:, 3]  # xmax
    y2 = dets[:, 4]  # ymax
    scores = dets[:, 5]  # confidence

    # 计算每个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # 按置信度分数降序排序

    for i in range(len(order)):
        idx = order[i]
        maxpos = idx
        maxscore = scores[idx]

        tx1, ty1, tx2, ty2, ts = x1[idx], y1[idx], x2[idx], y2[idx], scores[idx]

        # 当前bbox和剩下bbox之间的交叉区域
        xx1 = np.maximum(tx1, x1[order[i+1:]])
        yy1 = np.maximum(ty1, y1[order[i+1:]])
        xx2 = np.minimum(tx2, x2[order[i+1:]])
        yy2 = np.minimum(ty2, y2[order[i+1:]])

        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算iou
        ovr = inter / (areas[idx] + areas[order[i+1:]] - inter)

        # 使用高斯衰减函数或线性衰减函数
        if method == 1:
            weight = np.maximum(0.0, 1 - ovr)
        elif method == 2:
            weight = np.exp(-(ovr * ovr) / sigma)
        else:
            raise ValueError("method must be 1 (linear) or 2 (gaussian)")

        # 只对 scores[i + 1:] 中小于 (1+ovr)*thresh 的项进行权重衰减
        #score_mask = scores[order[i + 1:]] < (1 + ovr) * 0.5
        #scores[order[i + 1:]][score_mask] *= weight[score_mask]
        scores[order[i+1:]] *=weight
    # 筛选出大于阈值的detection
    keep = dets[scores > thresh]

    return keep

def compute_f(img1,img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 使用SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 根据匹配点的距离进行排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # 计算仿射变换矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H
def read_xml_r(xml_file1, i):
    # 读取xml文件
    root1 = ET.parse(xml_file1).getroot()  # xml文件根
    track_all = root1.findall('track')
    # 初始化box及其对应的ids
    bboxes1 = []
    ids = []
    id = 0
    for track in track_all:
        id_label = int(track.attrib['id'])
        # label = int(category[track.attrib['label']])
        boxes = track.findall('box')
        shape = [1080, 1920]
        for box in boxes:
            # print(box.attrib['frame'])
            if int(box.attrib['frame']) == i:
                xtl = int(box.attrib['xtl'])
                ytl = int(box.attrib['ytl'])
                xbr = int(box.attrib['xbr'])
                ybr = int(box.attrib['ybr'])
                outside = int(box.attrib['outside'])
                occluded = int(box.attrib['occluded'])
                centx1 = int((xtl + xbr) / 2)
                centy1 = int((ytl + ybr) / 2)
                #
                if outside == 1 or xtl <= 10 and ytl <= 10 or xbr >= shape[1] - 0 and ytl <= 0 \
                        or xtl <= 10 and ybr >= shape[0] - 10 or xbr >= shape[1] - 10 and ybr >= shape[0] - 10:
                    break
                # cv2.rectangle(image1, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
                # cv2.putText(image1, str(id), (xtl, ybr), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
                # cv2.imshow("jj", image1)
                # cv2.waitKey(10)
                confidence = 0.99
                bboxes1.append([xtl, ytl, xbr, ybr, 1])
                ids.append(id_label)
                id += 1
                break
    bboxes1 = torch.tensor(bboxes1, dtype=torch.long)
    ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.zeros_like(ids)
    return bboxes1, ids, labels


def calculate_cent_corner_pst(img1, result1):
    cent_allclass = []
    corner_allclass = []

    # for calss_num, result in enumerate(result1):
    center_pst = np.array([])
    corner_pst = np.array([])
    for dots in result1:
        # print("dots:", dots)
        x1 = dots[1]
        y1 = dots[2]
        x2 = dots[3]
        y2 = dots[4]
        centx = (x1 + x2) / 2
        centy = (y1 + y2) / 2
        # 收集检测结果的中点和角点
        if center_pst.size == 0:
            center_pst = np.array([[centx, centy]])
        else:
            center_pst = np.append(center_pst, [[centx, centy]], axis=0)  # 前缀pst=和axis = 0一定要有!!!!!!!!!!
        if corner_pst.size == 0:
            corner_pst = np.array([[x1, y1],
                                   [x2, y2]])
        else:
            corner_pst = np.append(corner_pst, [[x1, y1], [x2, y2]], axis=0)  # 前缀pst=和axis = 0一定要有!!!!!!!!!!
        # cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 33, 32), 5)
    # center_pst = center_pst.reshape(-1, 2).astype(np.float32)
    # corner_pst = corner_pst.reshape(-1,  2).astype(np.float32)

    # cent_allclass.append(center_pst)
    # corner_allclass.append(corner_pst)
    # cv2.imshow("img1", img1)
    # cv2.waitKey(100)

    return center_pst, corner_pst

def calculate_cent_corner_pst(img1, result1):
    cent_allclass = []
    corner_allclass = []

    # for calss_num, result in enumerate(result1):
    center_pst = np.array([])
    corner_pst = np.array([])
    for dots in result1:
        # print("dots:", dots)
        x1 = dots[1]
        y1 = dots[2]
        x2 = dots[3]
        y2 = dots[4]
        centx = (x1 + x2) / 2
        centy = (y1 + y2) / 2
        # 收集检测结果的中点和角点
        if center_pst.size == 0:
            center_pst = np.array([[centx, centy]])
        else:
            center_pst = np.append(center_pst, [[centx, centy]], axis=0)  # 前缀pst=和axis = 0一定要有!!!!!!!!!!
        if corner_pst.size == 0:
            corner_pst = np.array([[x1, y1],
                                   [x2, y2]])
        else:
            corner_pst = np.append(corner_pst, [[x1, y1], [x2, y2]], axis=0)  # 前缀pst=和axis = 0一定要有!!!!!!!!!!
        # cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 33, 32), 5)
    # center_pst = center_pst.reshape(-1, 2).astype(np.float32)
    # corner_pst = corner_pst.reshape(-1,  2).astype(np.float32)

    # cent_allclass.append(center_pst)
    # corner_allclass.append(corner_pst)
    # cv2.imshow("img1", img1)
    # cv2.waitKey(100)

    return center_pst, corner_pst



# 遍历两个trackbox,统计同ID:
def get_matched_ids_frame1(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2):
    matched_ids_cache = []
    pts_src = []
    pts_dst = []
    for m, dots in enumerate(track_bboxes):
        trac_id = dots[0]
        # A_max_id = trac_id if A_max_id<trac_id else A_max_id
        for n, dots2 in enumerate(track_bboxes2):
            trac2_id = dots2[0]
            # B_max_id = trac2_id if B_max_id<trac2_id else B_max_id
            if trac_id == trac2_id:
                # cv2.circle(image1, (int(cent_allclass[m][0]), int(cent_allclass[m][1])), 30, (0, 345, 255))
                # cv2.imshow('img', image1)
                # cv2.waitKey(100)
                # cv2.circle(image2, (int(cent_allclass2[n][0]), int(cent_allclass2[n][1])), 30, (54, 0, 255))
                # cv2.imshow('img2', image2)
                # cv2.waitKey(10)
                # 将匹双机配点中心放入列表里面，后续用来计算旋转矩阵
                pts_src.append(cent_allclass[m])
                pts_dst.append(cent_allclass2[n])
                matched_ids_cache.append(trac_id)
                break

    # 变换数据格式，用于后续计算变换矩阵
    pts_src = np.array(pts_src)
    pts_dst = np.array(pts_dst)
    matched_ids = matched_ids_cache.copy()
    return pts_src, pts_dst, matched_ids


def get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass, corner_allclass2,
                    A_max_id, B_max_id, coID_confirme):
    matched_ids_cache = []
    A_new_ID = []
    B_new_ID = []
    A_pts = []
    B_pts = []
    A_pts_corner = []
    B_pts_corner = []
    A_old_not_matched_ids = []
    B_old_not_matched_ids = []
    A_old_not_matched_pts = []
    B_old_not_matched_pts = []
    A_old_not_matched_pts_corner = []
    B_old_not_matched_pts_corner = []

    pts_src = []
    pts_dst = []
    for m, dots in enumerate(track_bboxes):
        trac_id = dots[0]
        if trac_id in coID_confirme:
            print("trac_id", trac_id)
            matched_ids_cache.append(trac_id)
            pts_src.append(cent_allclass[m])
            print("cent_allclass[m]", cent_allclass[m])
            for n, dots2 in enumerate(track_bboxes2):
                trac2_id = dots2[0]
                if trac2_id == trac_id:
                    pts_dst.append(cent_allclass2[n])
                    print("cent_allclass2[n]", cent_allclass2[n])
            continue
        if A_max_id < trac_id:
            if trac_id not in A_new_ID:
                A_new_ID.append(trac_id)
                A_pts.append(cent_allclass[m])
                A_pts_corner.append([corner_allclass[2 * m], corner_allclass[2 * m + 1]])
            continue
        for n, dots2 in enumerate(track_bboxes2):
            trac2_id = dots2[0]
            flag_matched = 0
            if B_max_id < trac2_id:
                if trac2_id not in B_new_ID:
                    B_new_ID.append(trac2_id)
                    B_pts.append(cent_allclass2[n])
                    B_pts_corner.append([corner_allclass2[2 * n], corner_allclass2[2 * n + 1]])
                continue
            if trac_id == trac2_id:
                # cv2.circle(image1, (int(cent_allclass[m][0]), int(cent_allclass[m][1])), 30, (0, 345, 255))
                # cv2.imshow('img', image1)
                # cv2.waitKey(100)
                # cv2.circle(image2, (int(cent_allclass2[n][0]), int(cent_allclass2[n][1])), 30, (54, 0, 255))
                # cv2.imshow('img2', image2)
                # cv2.waitKey(10)
                # 将匹双机配点中心放入列表里面，后续用来计算旋转矩阵
                matched_ids_cache.append(trac_id)
                pts_src.append(cent_allclass[m])
                pts_dst.append(cent_allclass2[n])
                flag_matched = 1
                break
        if flag_matched == 0 and trac_id not in A_old_not_matched_ids:
                A_old_not_matched_ids.append(trac_id)
                A_old_not_matched_pts.append(cent_allclass[m])
                A_old_not_matched_pts_corner.append([corner_allclass[2 * m], corner_allclass[2 * m + 1]])

    for n, dots2 in enumerate(track_bboxes2):
        trac2_id = dots2[0]
    if trac2_id not in B_old_not_matched_ids and trac2_id not in matched_ids_cache and trac2_id not in B_new_ID:
        B_old_not_matched_ids.append(trac2_id)
        B_old_not_matched_pts.append(cent_allclass2[n])
        B_old_not_matched_pts_corner.append([corner_allclass2[2 * n], corner_allclass2[2 * n + 1]])

            # else:  # A中旧目标没配对的也加入匹配序列
            #     if trac_id not in A_new_ID:
            #         A_new_ID.append(trac_id)
            #         A_pts.append(cent_allclass[m])
            #         A_pts_corner.append([corner_allclass[2 * m], corner_allclass[2 * m + 1]])
    matched_ids = matched_ids_cache.copy()
    # 变换数据格式，用于后续计算变换矩阵
    pts_src = np.array(pts_src)
    pts_dst = np.array(pts_dst)
    return matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, B_old_not_matched_pts_corner


def get_matched_det(cent_allclass_det,corner_allclass_det,det_bboxes,f,cent_allclass2_det, corner_allclass2_det,det_bboxes2,thres):#未配对的检测点被直接舍弃，可能可以考虑IOU（此处未考虑）

    det_boxes=[]
    det_boxes2=[]
    if len(cent_allclass_det)==0:
        pass
    else:
        cent_allclass_det= np.array(cent_allclass_det).reshape(-1, 1, 2).astype(np.float32)
        corner_allclass_det = np.array(corner_allclass_det).reshape(-1, 1, 2).astype(np.float32)
        cent_allclass_det=cv2.perspectiveTransform(cent_allclass_det,f)
        corner_allclass_det=cv2.perspectiveTransform(corner_allclass_det,f)
        if cent_allclass_det is not None:
            #dist=np.zeros((1,1))

            for ii,xy in enumerate(cent_allclass_det):

                min_x = min(corner_allclass_det[ii * 2, 0, 0], corner_allclass_det[ii * 2 + 1, 0, 0])
                max_x = max(corner_allclass_det[ii * 2, 0, 0], corner_allclass_det[ii * 2 + 1, 0, 0])
                min_y = min(corner_allclass_det[ii * 2, 0, 1], corner_allclass_det[ii * 2 + 1, 0, 1])
                max_y = max(corner_allclass_det[ii * 2, 0, 1], corner_allclass_det[ii * 2 + 1, 0, 1])
                dist = 100000
                ji = -1
                if min_x > 1920 or max_x < 0 or min_y > 1080 or max_y < 0 or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080:
                    continue
                else:
                    for j,dots in enumerate(cent_allclass2_det):
                        centx=dots[0]
                        centy=dots[1]
                        dist_1= ((xy[0, 0] - centx) ** 2 + (xy[0, 1] - centy) ** 2) ** 0.5
                        if dist_1<dist:
                            dist=dist_1
                            ji=j
                    if dist<thres:
                        det_boxes.append(det_bboxes[ii])
                        det_boxes2.append(det_bboxes2[ji])
    return det_boxes,det_boxes2

def get_matched_trackboxes(track_bboxes,track_bboxes2):
    track_boxes=[]
    track_boxes2=[]

    if len(track_bboxes)==0:
        pass
    else:
        for ii,xy in enumerate(track_bboxes):
            for jj,yz in enumerate(track_bboxes2):
                if xy[0]==yz[0]:
                    track_boxes.append(track_bboxes[ii])
                    track_boxes2.append(track_bboxes2[jj])
    return track_boxes,track_boxes2

def get_chosed_track(track_bboxes,track_bboxes2,threshold):
    track_boxes=[]
    track_boxes2=[]
    if len(track_bboxes)==0:
        pass
    else:
        for ii,xy in enumerate(track_bboxes):
            if xy[-1]>=threshold:
                track_boxes.append(xy)
        for jj,yz in enumerate(track_bboxes2):
            if yz[-1]>=threshold:
                track_boxes2.append(yz)
    track_boxes= np.array(track_boxes)
    track_boxes2= np.array(track_boxes2)
    return track_boxes,track_boxes2



def A_same_target_refresh_same_ID(A_new_ID_func, A_pts_func, A_pts_corner_func, f1_func, cent_allclass2_func,
                                  track_bboxes_func, track_bboxes2_func, matched_ids_func, det_bboxes, det_bboxes2,
                                  image2, coID_confirme, thres=100):
    flag = 0
    IOU_flag = 0
    if len(A_pts_func) == 0:
        pass
    else:
        A_pts_func = np.array(A_pts_func).reshape(-1, 1, 2).astype(np.float32)
        A_pts_corner_func = np.array(A_pts_corner_func).reshape(-1, 1, 2).astype(np.float32)
        A_dst_func = cv2.perspectiveTransform(A_pts_func, f1_func)
        A_dst_corner_func = cv2.perspectiveTransform(A_pts_corner_func, f1_func)
        # print(A_pts, A_dst)
        # zipped = zip(A_dst, A_dst_corner)
        # for ii_class, (dst_cent, dst_corner) in enumerate(list(zipped)):
        if A_dst_func is not None:
            dist = np.zeros((len(A_dst_func), len(track_bboxes2_func)))
            # 以下为了计算valid，并补充到检测结果中##################3
            valid = []
            for ii, xy in enumerate(A_dst_func):
                min_x = min(A_dst_corner_func[ii * 2, 0, 0], A_dst_corner_func[ii * 2 + 1, 0, 0])
                max_x = max(A_dst_corner_func[ii * 2, 0, 0], A_dst_corner_func[ii * 2 + 1, 0, 0])
                min_y = min(A_dst_corner_func[ii * 2, 0, 1], A_dst_corner_func[ii * 2 + 1, 0, 1])
                max_y = max(A_dst_corner_func[ii * 2, 0, 1], A_dst_corner_func[ii * 2 + 1, 0, 1])
                if min_x > 1920 or max_x < 0 or min_y > 1080 or max_y < 0 or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080:
                    continue
                else:
                    for j, dots in enumerate(cent_allclass2_func):  # track_bboxes2  ndarray n*6
                        centx = int(dots[0])
                        centy = int(dots[1])
                        # cv2.circle(img2, (centx, centy), 9, (150, 34, 23), 3)
                        # cv2.circle(img2, (int(xy[0, 0]), int(xy[0, 1])), 9, (50, 340, 23), 3)

                        dist[ii, j] = ((xy[0, 0] - centx) ** 2 + (xy[0, 1] - centy) ** 2) ** 0.5
                        # cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (33, 33, 32), 5)
                    # 超参数！35
                    # print(ii, dist)
                    if min(dist[ii], default=0) < thres:
                        # A:A_index   B:B_index
                        # A_index = list(track_bboxes[:, 0]).index(A_new_ID[ii])
                        # B_index = list(dist[ii]).index(min(dist[ii], default=0))
                        # print(np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0])
                        A_index = np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0]# [0].astype(int)
                        B_index = np.where(dist[ii] == min(dist[ii]))[0]
                        # print(A_index)
                        # print(B_index)
                        if len(A_index) > 1:
                            A_index = min(A_index)  # #####################??????????????????????????????????????????min???
                        else:
                            A_index = A_index[0]
                        if len(B_index) > 1:
                            B_index = min(B_index)  # #####################??????????????????????????????????????????min???
                        else:
                            B_index = B_index[0]
                        print(A_index, B_index)
                        if track_bboxes2_func[B_index, 0] not in matched_ids_func:
                            # max_id = max(track_bboxes2[B_index, 0], A_new_ID[ii])
                            min_id = int(min(track_bboxes2_func[B_index, 0], A_new_ID_func[ii]))
                            track_bboxes_func[A_index, 0] = min_id
                            track_bboxes2_func[B_index, 0] = min_id
                            # 最重要的一行语句：
                            print("step1: coID confirme: {0} {1} to {2}".format(track_bboxes_func[A_index, 0],
                                                                         track_bboxes2_func[B_index, 0], min_id))
                            coID_confirme.append(int(min_id))
                            matched_ids_func.append(int(min_id))

    return track_bboxes_func, track_bboxes2_func, matched_ids_func, flag, coID_confirme


def B_same_target_refresh_same_ID(B_new_ID, B_pts, B_pts_corner, f2, cent_allclass,
                                  track_bboxes, track_bboxes2, matched_ids, det_bboxes, det_bboxes2,
                                  image2, coID_confirme, thres=150):
    flag = 0
    if len(B_pts) == 0:
        pass
    else:
        B_pts = np.array(B_pts).reshape(-1, 1, 2).astype(np.float32)
        B_pts_corner = np.array(B_pts_corner).reshape(-1, 1, 2).astype(np.float32)
        B_dst = cv2.perspectiveTransform(B_pts, f2)
        B_dst_corner = cv2.perspectiveTransform(B_pts_corner, f2)
        # print(A_pts, A_dst)
        # zipped = zip(A_dst, A_dst_corner)
        # for ii_class, (dst_cent, dst_corner) in enumerate(list(zipped)):
        if B_pts is not None:
            dist = np.zeros((len(B_pts), len(track_bboxes)))
            # 以下为了计算valid，并补充到检测结果中##################3
            valid = []
            for ii, xy in enumerate(B_dst):
                min_x = min(B_dst_corner[ii * 2, 0, 0], B_dst_corner[ii * 2 + 1, 0, 0])
                max_x = max(B_dst_corner[ii * 2, 0, 0], B_dst_corner[ii * 2 + 1, 0, 0])
                min_y = min(B_dst_corner[ii * 2, 0, 1], B_dst_corner[ii * 2 + 1, 0, 1])
                max_y = max(B_dst_corner[ii * 2, 0, 1], B_dst_corner[ii * 2 + 1, 0, 1])
                if min_x > 1920 or max_x < 0 or min_y > 1080 or max_y < 0 or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080:
                    continue
                else:
                    for j, dots in enumerate(cent_allclass):  # track_bboxes2  ndarray n*6
                        centx = int(dots[0])
                        centy = int(dots[1])
                        # cv2.circle(img2, (centx, centy), 2, (ii_class*50,34,23), 3)
                        # cv2.circle(img2, (int(xy[0, 0]), int(xy[0, 1])), 2, (ii_class*50,34,23), 3)

                        dist[ii, j] = ((xy[0, 0] - centx) ** 2 + (xy[0, 1] - centy) ** 2) ** 0.5
                        # cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (33, 33, 32), 5)
                    # 超参数！35
                    # print(ii, dist)
                    if min(dist[ii], default=0) < thres:

                        # A:A_index   B:B_index
                        # A_index = list(track_bboxes[:, 0]).index(A_new_ID[ii])
                        # B_index = list(dist[ii]).index(min(dist[ii], default=0))
                        B_index = np.where(track_bboxes2[:, 0] == B_new_ID[ii])[0]
                        A_index = np.where(dist[ii] == min(dist[ii]))[0]
                        # print(A_index)
                        # print(B_index)
                        if len(A_index) > 1:
                            A_index = min(A_index)  # #####################??????????????????????????????????????????min???
                        else:
                            A_index = A_index[0]
                        if len(B_index) > 1:
                            B_index = min(B_index)  # #####################??????????????????????????????????????????min???
                        else:
                            B_index = B_index[0]
                        print(A_index, B_index)
                        # 需不需要加这个判断？？？？？？？？？？
                        if track_bboxes[A_index, 0] not in matched_ids:
                            # max_id = max(track_bboxes2[B_index, 0], A_new_ID[ii])
                            # print(track_bboxes[A_index, 0], B_new_ID[ii])
                            min_id = min(track_bboxes[A_index, 0], B_new_ID[ii])
                            track_bboxes[A_index, 0] = min_id
                            track_bboxes2[B_index, 0] = min_id
                            # 最重要的一行语句：
                            # print(min_id)
                            print("step2: coID confirme: {0} {1} to {2}".format(track_bboxes[A_index, 0],
                                                                         track_bboxes2[B_index, 0], min_id))
                            coID_confirme.append(int(min_id))
                            matched_ids.append(int(min_id))
                            # print("coID confirmeB_2_A:", min_id)
                        # else:
                    # else:
                    #     # 让映射框和检测框做匹配，匹配上的检测框做为补充框并且加入matched ids
                    #     for boxx in det_bboxes:
                    #         xmin1, ymin1, xmax1, ymax1 = min_x, min_y, max_x, max_y
                    #         xmin2, ymin2, xmax2, ymax2 = boxx[0], boxx[1], boxx[2], boxx[3]
                    #         xx1 = np.max([xmin1, xmin2])
                    #         yy1 = np.max([ymin1, ymin2])
                    #         xx2 = np.min([xmax1, xmax2])
                    #         yy2 = np.min([ymax1, ymax2])
                    #
                    #         area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                    #         area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                    #         inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                    #         iou = inter_area / (area1 + area2 - inter_area + 1e-6)
                    #         if iou >= 0.6:
                    #             print("supply A")
                    #             track_bboxes = np.concatenate((track_bboxes, np.array([[B_new_ID[ii], xmin2, ymin2, xmax2, ymax2, 0.999]])), axis=0)
                    #             matched_ids.append(int(B_new_ID[ii]))
                    # np.append(track_bboxes, [B_new_ID[ii], min_x, min_y, max_x, max_y, 0.999])

                    # valid.append([min(dst_corner[ii * 2, 0, 0], dst_corner[ii * 2 + 1, 0, 0]),
                    #               min(dst_corner[ii * 2, 0, 1], dst_corner[ii * 2 + 1, 0, 1]),
                    #               max(dst_corner[ii * 2, 0, 0], dst_corner[ii * 2 + 1, 0, 0]),
                    #               max(dst_corner[ii * 2, 0, 1], dst_corner[ii * 2 + 1, 0, 1]),
                    #               0.99])  # 左上右下点的确定
                    # cv2.rectangle(img2, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 33, 32), 5)

    return track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme


def same_target_refresh_same_ID(A_new_ID_func, A_pts_func, A_pts_corner_func, f1_func, cent_allclass2_func,
                                track_bboxes_func, track_bboxes2_func, matched_ids_func, det_bboxes, det_bboxes2,
                                image2, coID_confirme, thres=100):
    flag = 0
    IOU_flag = 0
    if len(A_pts_func) == 0:
        pass
    else:
        A_pts_func = np.array(A_pts_func).reshape(-1, 1, 2).astype(np.float32)
        A_pts_corner_func = np.array(A_pts_corner_func).reshape(-1, 1, 2).astype(np.float32)
        A_dst_func = cv2.perspectiveTransform(A_pts_func, f1_func)
        A_dst_corner_func = cv2.perspectiveTransform(A_pts_corner_func, f1_func)
        # print(A_pts, A_dst)
        # zipped = zip(A_dst, A_dst_corner)
        # for ii_class, (dst_cent, dst_corner) in enumerate(list(zipped)):
        if A_dst_func is not None:
            dist = np.zeros((len(A_dst_func), len(track_bboxes2_func)))
            # 以下为了计算valid，并补充到检测结果中##################3
            valid = []
            for ii, xy in enumerate(A_dst_func):
                min_x = min(A_dst_corner_func[ii * 2, 0, 0], A_dst_corner_func[ii * 2 + 1, 0, 0])
                max_x = max(A_dst_corner_func[ii * 2, 0, 0], A_dst_corner_func[ii * 2 + 1, 0, 0])
                min_y = min(A_dst_corner_func[ii * 2, 0, 1], A_dst_corner_func[ii * 2 + 1, 0, 1])
                max_y = max(A_dst_corner_func[ii * 2, 0, 1], A_dst_corner_func[ii * 2 + 1, 0, 1])
                # or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080这个判断是后来加的，因为图像在边界的标注有问题！！！！！！！！！！！！！！！
                if min_x > 1920 or max_x < 0 or min_y > 1080 or max_y < 0 or min_x < 0 or min_y < 0 or max_x > 1920 or max_y > 1080:
                    continue
                else:
                    for j, dots in enumerate(cent_allclass2_func):  # track_bboxes2  ndarray n*6
                        centx = int(dots[0])
                        centy = int(dots[1])
                        # cv2.circle(img2, (centx, centy), 9, (150, 34, 23), 3)
                        # cv2.circle(img2, (int(xy[0, 0]), int(xy[0, 1])), 9, (50, 340, 23), 3)

                        dist[ii, j] = ((xy[0, 0] - centx) ** 2 + (xy[0, 1] - centy) ** 2) ** 0.5
                        # cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (33, 33, 32), 5)
                    # 超参数！35
                    # print(ii, dist)
                    if min(dist[ii], default=0) < thres:
                        # A:A_index   B:B_index
                        # A_index = list(track_bboxes[:, 0]).index(A_new_ID[ii])
                        # B_index = list(dist[ii]).index(min(dist[ii], default=0))
                        # print(np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0])
                        A_index = np.where(track_bboxes_func[:, 0] == A_new_ID_func[ii])[0]  # [0].astype(int)
                        B_index = np.where(dist[ii] == min(dist[ii]))[0]
                        if len(A_index) > 1:
                            A_index = min(
                                A_index)  # #####################??????????????????????????????????????????min???
                        else:
                            A_index = A_index[0]
                        if len(B_index) > 1:
                            B_index = min(
                                B_index)  # #####################??????????????????????????????????????????min???
                        else:
                            B_index = B_index[0]
                        # if track_bboxes2_func[B_index, 0] in matched_ids_func:  # 比较大小取小

                        if track_bboxes2_func[B_index, 0] not in matched_ids_func:
                            ##################matchede id里面还可能有错误匹配，进行进一步激励计算取最近的加入matched id,另一个重新加入unmatched ids，在后续进行rematch
                            ##################matchede id里面还可能有错误匹配，进行进一步激励计算取最近的加入matched id,另一个重新加入unmatched ids，在后续进行rematch
                            ##################matchede id里面还可能有错误匹配，进行进一步激励计算取最近的加入matched id,另一个重新加入unmatched ids，在后续进行rematch
                            # max_id = max(track_bboxes2[B_index, 0], A_new_ID[ii])
                            min_id = int(min(track_bboxes2_func[B_index, 0], A_new_ID_func[ii]))
                            track_bboxes_func[A_index, 0] = min_id
                            track_bboxes2_func[B_index, 0] = min_id
                            # 最重要的一行语句：
                            # print("coID confirme:", min_id)
                            print("step3: coID confirme: {0} {1} to {2}".format(track_bboxes_func[A_index, 0],
                                                                         track_bboxes2_func[B_index, 0], min_id))
                            coID_confirme.append(int(min_id))
                            matched_ids_func.append(int(min_id))
                    # else:
                    #     IOU_flag = 0
                    #     iou = np.array([])
                    #     # 让映射框和检测框做匹配，匹配上的检测框做为补充框并且加入matched ids
                    #     for boxx in det_bboxes2:
                    #         # 保证补充狂在图像内，防止溢出边界
                    #         min_x = min_x if min_x > 0 else 0
                    #         min_y = min_y if min_y > 0 else 0
                    #         max_x = max_x if max_x < 1920 else 1920
                    #         max_y = max_y if max_y < 1080 else 1080
                    #         # 下面是计算两个框IOU的程序：
                    #         xmin1, ymin1, xmax1, ymax1 = min_x, min_y, max_x, max_y
                    #         xmin2, ymin2, xmax2, ymax2 = boxx[0], boxx[1], boxx[2], boxx[3]
                    #
                    #         xx1 = np.max([xmin1, xmin2])
                    #         yy1 = np.max([ymin1, ymin2])
                    #         xx2 = np.min([xmax1, xmax2])
                    #         yy2 = np.min([ymax1, ymax2])
                    #
                    #         area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                    #         area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                    #         inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                    #         iou = np.append(iou, inter_area / (area1 + area2 - inter_area + 1e-6))
                    #         # 如果两个框IOU大于一定阈值则进行检测框补充，否则直接进行映射狂补充
                    #     if max(iou) > 0.3:
                    #         # indexindex = np.where(iou == max(iou))[0].astype(int)
                    #         # boxx = det_bboxes2[indexindex][0]
                    #         # 选置信度高的
                    #         indexindex = np.where(iou > 0.3)
                    #
                    #         if len(indexindex[0]) == 1:
                    #             indexindex = np.where(iou == max(iou))[0].astype(int)
                    #             boxx = det_bboxes2[indexindex][0]
                    #         else:
                    #             print(max(det_bboxes2[indexindex][:, -1]))
                    #             index = np.where(
                    #                 np.array(det_bboxes2[indexindex][:, -1]) == max(det_bboxes2[indexindex][:, -1]))
                    #             # print(indexindex)
                    #             # print(index[0])
                    #             # print(indexindex[0][index[0]])
                    #             boxx = det_bboxes2[indexindex[0][index[0]]][0]
                    #         # print(boxx)
                    #         xmin2, ymin2, xmax2, ymax2 = boxx[0], boxx[1], boxx[2], boxx[3]
                    #         if (xmax2 - xmin2) < 20:
                    #             continue
                    #         if (ymax2 - ymin2) < 20:
                    #             continue
                    #         print("suppliment")
                    #         IOU_flag = 1
                    #         track_bboxes2_func = np.concatenate((track_bboxes2_func, np.array(
                    #             [[A_new_ID_func[ii], xmin2, ymin2, xmax2, ymax2, 0.999]])),
                    #                                             axis=0)  # !!!track_bboxes2_func = !!!
                    #         matched_ids_func.append(A_new_ID_func[ii])
                    #         flag = 1
                    #         print("flag == 1")
                    #         cv2.rectangle(image2, (int(xmin2), int(ymin2)), (int(xmax2), int(ymax2)), (250, 33, 32), 5)
                    #         cv2.imshow("fksoadf", image2)
                    #         cv2.waitKey(100)
                    #         flag = 0
                    #         # print("new id", A_new_ID_func[ii])
                    #         # print(track_bboxes2_func[-1])

                        # 对于置信度高的可以直接映射过去
                        # if IOU_flag == 0:
                        #     print("directly_suppliment")
                        #     track_bboxes2_func = np.concatenate((track_bboxes2_func, np.array(
                        #         [[A_new_ID_func[ii], min_x, min_y, max_x, max_y, 0.999]])), axis=0)
                        #     matched_ids_func.append(A_new_ID_func[ii])
                        #     flag = 1
                        #     print("flag == 1")
                        #     cv2.rectangle(image2, (int(min_x), int(min_y)),
                        #                   (int(max_x), int(max_y)), (250, 33, 32), 5)
                        #     cv2.imshow("fksoadf", image2)
                        #     cv2.waitKey(1000)
                        #     flag = 0

    return track_bboxes_func, track_bboxes2_func, matched_ids_func, flag, coID_confirme
