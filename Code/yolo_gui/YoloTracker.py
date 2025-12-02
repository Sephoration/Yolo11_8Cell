import cv2
import torch
from ultralytics import YOLO
from yolo_gui.baseDetect import baseDetect

# 只关注的类别（侦测 & 追踪共用）
OBJ_LIST = ['person', 'car', 'bus', 'truck']

class yoloTracker(baseDetect):
    """
    统一的 YOLO 类：
    - detect(im) : 纯目标侦测（不带 track_id）
    - track(im)  : 目标侦测 + 多目标追踪（带 track_id）
    """

    def __init__(self, model_path):
        super().__init__()
        self.weights = model_path
        self.init_model()

    def init_model(self):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.names = self.model.names

    # ---------------------------------------------------------
    # 1. 纯目标侦测（Detection）
    # ---------------------------------------------------------
    def detect(self, im):
        """
        纯目标侦测（不带追踪 ID）。

        参数
        ----
        im : np.ndarray (BGR)
            输入图像（OpenCV 读取的一帧）

        返回
        ----
        im_out : np.ndarray (BGR)
            画好框的图像
        pred_boxes : list of tuple
            每个元素为 (x1, y1, x2, y2, lbl, confidence, track_id)
            此处 track_id 统一为 None，方便与 track() 结果保持结构一致。
        """
        results = self.model.predict(
            source=im,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device
        )[0]

        pred_boxes = []
        for box in results.boxes:
            class_id = int(box.cls.cpu().item())
            lbl = self.names[class_id]

            # 只保留关心的类别
            if lbl not in OBJ_LIST:
                continue

            xyxy = box.xyxy.cpu()
            x1, y1, x2, y2 = xyxy[0].numpy()
            confidence = float(box.conf.cpu().item())

            track_id = None  # 侦测模式没有追踪 ID
            pred_boxes.append((x1, y1, x2, y2, lbl, confidence, track_id))

        # 使用基类的画框方法
        im_out = self.draw_bboxes(im, pred_boxes)
        return im_out, pred_boxes

    # ---------------------------------------------------------
    # 2. 目标追踪（Tracking）
    # ---------------------------------------------------------
    def track(self, im):
        """
        目标追踪：使用 YOLO + ByteTrack。

        参数
        ----
        im : np.ndarray (BGR)
            输入图像（OpenCV 读取的一帧）

        返回
        ----
        im_out : np.ndarray (BGR)
            画好框的图像
        pred_boxes : list of tuple
            每个元素为 (x1, y1, x2, y2, lbl, confidence, track_id)
        """
        results = self.model.track(
            im,
            tracker="bytetrack.yaml",
            persist=True,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device
        )

        detected_boxes = results[0].boxes
        pred_boxes = []

        for box in detected_boxes:
            class_id = int(box.cls.cpu().item())
            lbl = self.names[class_id]

            # 只保留关心的类别
            if lbl not in OBJ_LIST:
                continue

            xyxy = box.xyxy.cpu()
            x1, y1, x2, y2 = xyxy[0].numpy()
            confidence = float(box.conf.cpu().item())

            # track_id 对于追踪非常关键，如果没有则设为 None
            if box.id is None:
                track_id = None
            else:
                track_id = int(box.id.cpu().item())

            pred_boxes.append((x1, y1, x2, y2, lbl, confidence, track_id))

        # 使用基类的画框方法
        im_out = self.draw_bboxes(im, pred_boxes)
        return im_out, pred_boxes


# ---------------------------------------------------------
# 简单的独立测试入口（可选）
# ---------------------------------------------------------
if __name__ == '__main__':
    tracker = yoloTracker("../models/yolo11n.pt")

    # 测试 1：单张图片，纯侦测
    img_bgr = cv2.imread('../images/bus.jpg')
    if img_bgr is None:
        raise FileNotFoundError("Image not found: ../images/bus.jpg")

    det_img, det_boxes = tracker.detect(img_bgr.copy())
    cv2.imshow('Detection result', det_img)

    # 测试 2：同一张图片，追踪模式（单帧看起来和侦测类似，多了 ID 字段）
    track_img, track_boxes = tracker.track(img_bgr.copy())
    cv2.imshow('Tracking result', track_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
