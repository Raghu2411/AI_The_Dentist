import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.ops as ops
from torchvision import transforms as T
from ultralytics import YOLO
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
# from mmdet.apis import init_detector, inference_detector
from detectron2 import model_zoo
import config_master as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Detection utilities loading. Using device: {device}")


def load_all_models():
    """Loads all 5 models into a dictionary for the app to hold in memory."""
    models = {}
    try:
        models['yolov9'] = load_yolo(config.MODEL_PATHS['yolov9'])
        models['yolov11'] = load_yolo(config.MODEL_PATHS['yolov11'])
        models['faster_rcnn'] = load_torchvision_model('faster_rcnn', config.MODEL_PATHS['faster_rcnn'], config.NUM_CLASSES)
        models['retinanet'] = load_torchvision_model('retinanet', config.MODEL_PATHS['retinanet'], config.NUM_CLASSES)
        models['detectron2'] = load_detectron2(config.CONFIG_PATHS['detectron2'], config.MODEL_PATHS['detectron2'])
        # models['mmdetection'] = load_mmdetection(config.CONFIG_PATHS['mmdetection'], config.MODEL_PATHS['mmdetection'])
        print("\n All models loaded successfully")
    except Exception as e:
        print(f"Could not load all models.")
        print(f"Error: {e}")
        print("Please check all model paths.")
    return models

def load_yolo(weights_path):
    print(f"Loading YOLOv11 from {weights_path}...")
    return YOLO(weights_path)

def load_torchvision_model(model_name, weights_path, num_classes):
    print(f"Loading {model_name} from {weights_path}...")
    if model_name == 'faster_rcnn':
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
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

def load_detectron2(config_file_name, weights_path):
    """
    Loads Detectron2 model.
    config_file_name is a "model zoo name" (e.g., "COCO-InstanceSegmentation/...")
    """
    print(f"Loading Detectron2 from {weights_path}...")
    cfg = get_cfg()
    config_path = model_zoo.get_config_file(config_file_name)
    
    cfg.merge_from_file(config_path)
    
    cfg.MODEL.WEIGHTS = weights_path
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.NUM_CLASSES 
    
    return DefaultPredictor(cfg)

# def load_detectron2(config_file, weights_path):
#     print(f"Loading Detectron2 from {weights_path}...")
#     cfg = get_cfg()
#     cfg.merge_from_file(config_file)
#     cfg.MODEL.WEIGHTS = weights_path
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
#     cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#     return DefaultPredictor(cfg)

# def load_mmdetection(config_file, weights_path):
#     print(f"Loading MMDetection from {weights_path}...")
#     return init_detector(config_file, weights_path, device=device)

def _standardize_output(boxes, scores, class_ids, model_source):
    predictions = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        if class_id >= len(config.CLASSES): 
            continue 
        predictions.append({
            "bbox": box.tolist(),
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
    return _standardize_output(boxes, scores, class_ids, 'yolov11')

def run_detectron2_inference(predictor, image_cv2):
    outputs = predictor(image_cv2)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    class_ids = instances.pred_classes.numpy().astype(int)
    return _standardize_output(boxes, scores, class_ids, 'detectron2')

# def run_mmdetection_inference(model, image_cv2):
#     result = inference_detector(model, image_cv2)
#     pred_instances = result.pred_instances
#     boxes = pred_instances.bboxes.cpu().numpy()
#     scores = pred_instances.scores.cpu().numpy()
#     class_ids = pred_instances.labels.cpu().numpy().astype(int)
#     return _standardize_output(boxes, scores, class_ids, 'mmdetection')

def run_torchvision_inference(model, image_pil, model_source):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_pil).to(device)
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    class_ids = prediction['labels'].cpu().numpy().astype(int) 
    
    if model_source == 'faster_rcnn':
        valid_indices = class_ids > 0
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices] - 1
         
    return _standardize_output(boxes, scores, class_ids, model_source)

# Post-Processing Functions

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def filter_by_confidence(all_predictions):
    filtered_preds = []
    for pred in all_predictions:
        class_name = pred['class_name']
        threshold = config.CLASS_THRESHOLDS.get(class_name, 0.3)
        if pred['score'] >= threshold:
            filtered_preds.append(pred)
    return filtered_preds

def apply_batched_nms(predictions, iou_threshold):
    if not predictions:
        return []
    boxes = torch.tensor([p['bbox'] for p in predictions], dtype=torch.float32)
    scores = torch.tensor([p['score'] for p in predictions], dtype=torch.float32)
    class_idxs = torch.tensor([p['class_id'] for p in predictions], dtype=torch.long)
    keep_indices = ops.batched_nms(boxes, scores, class_idxs, iou_threshold)
    return [predictions[i] for i in keep_indices]

def deduplicate_teeth_by_score(teeth_list):
    """
    Ensures only one detection per FDI tooth class remains.
    Keeps the one with the highest confidence score.
    """
    best_teeth = {} # K: class_name, V: prediction_dict
    for tooth in teeth_list:
        class_name = tooth['class_name']
        score = tooth['score']
        
        if class_name not in best_teeth:
            best_teeth[class_name] = tooth
        else:
            if score > best_teeth[class_name]['score']:
                best_teeth[class_name] = tooth # Replace the old one with new one
                
    return list(best_teeth.values())

def assign_diseases_to_teeth(final_predictions):
    teeth = []
    diseases = []
    unassigned_diseases = []

    for pred in final_predictions:
        if pred['class_name'] in config.FDI_CLASSES:
            teeth.append(pred)
        elif pred['class_name'] == 'implant':
            unassigned_diseases.append(pred)
        else:
            diseases.append(pred)
    
    unique_teeth = deduplicate_teeth_by_score(teeth)
    structured_teeth = []
    for tooth in unique_teeth:
        tooth_dict = {**tooth, "conditions": []}
        for i, disease in enumerate(diseases):
            if disease.get('assigned', False):
                continue
            iou = calculate_iou(tooth['bbox'], disease['bbox'])
            if iou >= config.ASSIGNMENT_IOU_THRESHOLD:
                tooth_dict['conditions'].append(disease)
                diseases[i]['assigned'] = True
        structured_teeth.append(tooth_dict)

    unassigned = [d for d in diseases if not d.get('assigned', False)]
    unassigned.extend(unassigned_diseases)
    
    return structured_teeth, unassigned