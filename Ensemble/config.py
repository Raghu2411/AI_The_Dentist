import os

BASE_DIR = '/home/tq24401/llm/filtered_opg_dataset_for_LLM'
TEST_IMG_DIR = os.path.join(BASE_DIR, 'test')
TEST_JSON_PATH = os.path.join(BASE_DIR, 'test/_annotations.coco.json')

OUTPUT_DIR = './ensemble_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATHS = {
    'yolov9': '/home/tq24401/Dental/Detections/final_yolo11m.pt',
    'faster_rcnn': '/home/tq24401/Dental/Detections/final_faster_rcnn.pth',
    'retinanet': '/home/tq24401/Dental/Detections/final_retinanet.pth',
    'detectron2': '/home/tq24401/Dental/Detections/output_detectron2_maskrcnn/model_final_det.pth',
    'mmdetection': '/home/tq24401/Dental/Detections/work_dirs/mask_dino_dental/best_coco_bbox_mAP_epoch_10.pth'
}
CONFIG_PATHS = {
    'detectron2': '/home/tq24401/Dental/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'mmdetection': '/home/tq24401/Dental/Detections/mmdetection/configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py'
}

# Class Definitions (41 CLASSES)
CLASSES = [
    '11', '12', '13', '14', '15', '16', '17', '18', # 0-7
    '21', '22', '23', '24', '25', '26', '27', '28', # 8-15
    '31', '32', '33', '34', '35', '36', '37', '38', # 16-23
    '41', '42', '43', '44', '45', '46', '47', '48', # 24-31
    'calculus', 'caries', 'crown', 'impacted', 'implant', # 32-36
    'periapical radiolucency', 'rc-treated', 'restoration', 'root-stump' # 37-40
]

FDI_CLASSES = set(CLASSES[:32])
DISEASE_CLASSES = set(CLASSES[32:])
NUM_CLASSES = len(CLASSES) # 41

# Class-specific Confidence Thresholds
CLASS_THRESHOLDS = {
    **{fdi_class: 0.6 for fdi_class in FDI_CLASSES}, 
    'calculus': 0.35, 'caries': 0.18, 'crown': 0.5, 'impacted': 0.3, 
    'implant': 0.5, 'periapical radiolucency': 0.26, 'rc-treated': 0.5,
    'restoration': 0.5, 'root-stump': 0.4
}
NMS_IOU_THRESHOLD = 0.4 # Overlapping (Same Class)
ASSIGNMENT_IOU_THRESHOLD = 0.3 # Tooth vs Disease Pair

FDI_COLOR = (0, 1, 0) # Green
DISEASE_COLORS = {
    'calculus': (1, 0.5, 0),    # Orange
    'caries': (1, 0, 0),        # Red
    'crown': (0, 1, 1),         # Cyan
    'impacted': (0.8, 0.6, 1),  # Lavender
    'implant': (0, 0, 1),       # Blue
    'periapical radiolucency': (1, 1, 0), # Yellow
    'rc-treated': (0.5, 0, 0.5),  # Purple
    'restoration': (1.0, 0.843, 0.0), # gold
    'root-stump': (1, 0.8, 0.8)   # Pink
}
