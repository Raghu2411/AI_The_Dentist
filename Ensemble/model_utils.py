import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
from ultralytics import YOLO
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from mmdet.apis import init_detector, inference_detector
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loaders

def load_yolo(weights_path):
    print(f"Loading YOLOv9 from {weights_path}...")
    model = YOLO(weights_path)
    return model

def load_torchvision_model(model_name, weights_path, num_classes):
    print(f"Loading {model_name} from {weights_path}...")
    if model_name == 'faster_rcnn':
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # +1 for background class
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1) 
    else:
        model = retinanet_resnet50_fpn_v2(weights=None)
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_detectron2(config_file, weights_path):
    print(f"Loading Detectron2 from {weights_path}...")
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # Get all predictions
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = DefaultPredictor(cfg)
    return predictor

def load_mmdetection(config_file, weights_path):
    print(f"Loading MMDetection from {weights_path}...")
    model = init_detector(config_file, weights_path, device=device)
    return model

# Inference Runners

def _standardize_output(boxes, scores, class_ids, model_source):
    """Helper to create the standard list format."""
    predictions = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        # Check if class_id is valid for config list
        if class_id >= len(config.CLASSES): 
            continue 
        predictions.append({
            "bbox": box.tolist(), # [x1, y1, x2, y2]
            "score": float(score),
            "class_id": int(class_id),
            "class_name": config.CLASSES[int(class_id)],
            "model_source": model_source
        })
    return predictions

def run_yolo_inference(model, image_cv2):
    results = model(image_cv2, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    return _standardize_output(boxes, scores, class_ids, 'yolov9')

def run_detectron2_inference(predictor, image_cv2):
    outputs = predictor(image_cv2)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    class_ids = instances.pred_classes.numpy().astype(int)
    return _standardize_output(boxes, scores, class_ids, 'detectron2')

def run_mmdetection_inference(model, image_cv2):
    result = inference_detector(model, image_cv2)
    pred_instances = result.pred_instances
    boxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    class_ids = pred_instances.labels.cpu().numpy().astype(int)
    return _standardize_output(boxes, scores, class_ids, 'mmdetection')

def run_torchvision_inference(model, image_pil, model_source):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_pil).to(device)
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    class_ids = prediction['labels'].cpu().numpy().astype(int) 
    
    # Adjust for Faster R-CNN background class if it's 0-indexed
    if model_source == 'faster_rcnn':
        # Find all indices where class_id is not 0 (background)
        valid_indices = class_ids > 0
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices] - 1 # Shift all classes down by 1
         
    return _standardize_output(boxes, scores, class_ids, model_source)