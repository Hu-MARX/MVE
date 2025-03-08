import os
import numpy as np


# 加载目录中的所有检测框文件
# 加载目录中的所有检测框文件
def load_boxes_from_directory(directory_path):
    all_boxes = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  # 只处理 .txt 文件
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                boxes = load_boxes(file_path)
                for frame_id, box_list in boxes.items():
                    if frame_id not in all_boxes:
                        all_boxes[frame_id] = []
                    all_boxes[frame_id].extend(box_list)
    return all_boxes


# 加载单个文件中的检测框
def load_boxes(file_path):
    boxes = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split(',')))
            frame_id = int(parts[0])
            box = {
                'id': int(parts[1]),  # ID编号
                'x1': parts[2],  # 左上角X坐标
                'y1': parts[3],  # 左上角Y坐标
                'width': parts[4],  # 宽度
                'height': parts[5],  # 高度
                'area': (parts[4]) * (parts[5]),  # 面积
            }
            if frame_id not in boxes:
                boxes[frame_id] = []
            boxes[frame_id].append(box)
    return boxes


# 计算IoU
def calculate_iou(box1, box2):
    x1_max = max(box1['x1'], box2['x1'])
    y1_max = max(box1['y1'], box2['y1'])
    x2_min = min(box1['x1'] + box1['width'], box2['x1'] + box2['width'])
    y2_min = min(box1['y1'] + box1['height'], box2['y1'] + box2['height'])

    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    union = box1['area'] + box2['area'] - intersection
    return intersection / union if union > 0 else 0


# 计算AP
def calculate_ap(precisions, recalls):
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    indices = np.argsort(recalls)
    precisions = precisions[indices]
    recalls = recalls[indices]

    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])


# 计算 mAP
def calculate_map(detections, ground_truths, area_threshold, iou_threshold=0.5):
    aps = []
    for class_id in set([box['id'] for frame in ground_truths.values() for box in frame]):
        true_positives = []
        num_ground_truths = 0

        for frame_id in detections:
            if frame_id not in ground_truths:
                continue

            ground_truth_boxes = [box for box in ground_truths[frame_id] if box['id'] == class_id]
            detection_boxes = [box for box in detections[frame_id] if
                               box['id'] == class_id and box['area'] < area_threshold]
            num_ground_truths += len(ground_truth_boxes)

            matched = []
            for det_box in detection_boxes:
                best_iou = 0
                best_gt_box = None

                for gt_box in ground_truth_boxes:
                    iou = calculate_iou(det_box, gt_box)
                    if iou > best_iou and iou >= iou_threshold and gt_box not in matched:
                        best_iou = iou
                        best_gt_box = gt_box

                if best_gt_box is not None:
                    true_positives.append(1)
                    matched.append(best_gt_box)
                else:
                    true_positives.append(0)

        if num_ground_truths == 0:
            continue

        precisions = []
        recalls = []
        tp_sum = 0

        for i in range(len(true_positives)):
            tp_sum += true_positives[i]
            precision = tp_sum / (i + 1)
            recall = tp_sum / num_ground_truths
            precisions.append(precision)
            recalls.append(recall)

        aps.append(calculate_ap(precisions, recalls))

    return np.mean(aps) if aps else 0


# 加载数据
detections = load_boxes_from_directory('./test1')
ground_truths = load_boxes_from_directory('./txt/gt_true')

# 设置面积阈值
area_threshold = 500  # 可根据需要调整

# 计算 mAP-50 for small boxes
mAP_50 = calculate_map(detections, ground_truths, area_threshold, iou_threshold=0.5)
print(f'mAP-50 for small boxes (area < {area_threshold}): {mAP_50:.4f}')
