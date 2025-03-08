from ultralytics import YOLO
import torch
import numpy as np
import cv2

class Detector:
    def __init__(self, img_size=(1920, 1080)):
        self.img_size = img_size  # 设置为 (宽, 高)
        self.init_model()

    def init_model(self):
        self.weights = 'weights/best2.pt'  # 使用 YOLOv8 权重
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(self.weights)  # 加载 YOLOv8 模型
        self.model.to(self.device)  # 将模型移动到设备

    def preprocess(self, img):
        img0 = img.copy()
        img = cv2.resize(img, self.img_size)  # 调整大小
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0  # 转换为张量
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def detect(self, img):
        im0, _ = self.preprocess(img)
        results = self.model(img)  # 使用 YOLOv8 进行推理
        pred_boxes = []

        # 解析 YOLOv8 结果
        category = {'car': 3, 'bus': 2, 'person': 1, 'bicycle': 4}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 提取边界框
                cls = int(box.cls.cpu().numpy())  # 提取类别
                conf = float(box.conf.cpu().numpy()) # 提取置信度
                lbl = self.model.names[cls]

                if lbl in category:  # 判断是否在category字典中
                    lbl = category[lbl]  # 使用category中的数值替代标签
                    pred_boxes.append((x1, y1, x2, y2, conf, lbl))
        pred_boxes = np.array(pred_boxes)
        return im0, pred_boxes



class Detector_s:
    def __init__(self, img_size=(640,640)):
        self.img_size = img_size  # 设置为 (宽, 高)
        self.init_model()

    def init_model(self):
        self.weights = 'weights2/best.pt'  # 使用 YOLOv8 权重
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(self.weights)  # 加载 YOLOv8 模型
        self.model.to(self.device)  # 将模型移动到设备

    def preprocess(self, img):
        img0 = img.copy()
        img = cv2.resize(img, self.img_size)  # 调整大小
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0  # 转换为张量
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def detect(self, img):
        im0, _ = self.preprocess(img)
        results = self.model(img)  # 使用 YOLOv8 进行推理
        pred_boxes = []

        # 解析 YOLOv8 结果
        category = {'car': 3, 'bus': 2, 'person': 1, 'bicycle': 4}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 提取边界框
                cls = int(box.cls.cpu().numpy())  # 提取类别
                conf = float(box.conf.cpu().numpy()) # 提取置信度
                lbl = self.model.names[cls]

                if lbl in category:  # 判断是否在category字典中
                    lbl = category[lbl]  # 使用category中的数值替代标签
                    pred_boxes.append((x1, y1, x2, y2, conf, lbl))
        pred_boxes = np.array(pred_boxes)
        return im0, pred_boxes

