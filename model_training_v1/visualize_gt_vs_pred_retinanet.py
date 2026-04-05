"""
RetinaNet Visualization
Ground Truth vs Predictions with TP/FP/FN colors
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def get_system_font(size=32):
    """Get system font for visualization"""
    font_candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for f in font_candidates:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size)
            except:
                continue

    print("⚠ No system font found, fallback to default.")
    return ImageFont.load_default()


def compute_iou(boxA, boxB):
    """Compute IoU between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


@torch.no_grad()
def visualize_gt_vs_pred_retinanet(model_path, base_path, device, output_dir, 
                                   score_thresh=0.25, iou_thresh=0.5, 
                                   preserved_classes=None):
    """
    Visualize GT vs predictions with TP/FP/FN colors
    
    Args:
        model_path: Path to trained RetinaNet model
        base_path: Base directory containing test data
        device: Device to run on
        output_dir: Output directory for visualizations
        score_thresh: Confidence threshold for predictions
        iou_thresh: IoU threshold for matching
        preserved_classes: List of preserved class names (optional)
    """

    if preserved_classes is None:
        preserved_classes = ['calculus', 'caries', 'crown', 'impacted', 'implant',
                           'periapical radiolucency', 'rc-treated', 'restoration', 'root-stump']
    
    DETECTION_RESULTS = []

    print("\n Generating GT vs Prediction (TP/FP/FN colors)...\n")

    test_json = os.path.join(base_path, "test/_annotations.coco.json")
    test_img_dir = os.path.join(base_path, "test/images")  # Images directly in test/
    
    with open(test_json, "r") as f:
        coco_data = json.load(f)

    class_names = [c["name"] for c in sorted(coco_data["categories"], key=lambda x: x["id"])]
    preserved_ids = [class_names.index(c) for c in preserved_classes if c in class_names]

    TP_COLOR = (0, 255, 0)      # Green - True Positive
    FP_COLOR = (255, 0, 0)      # Red - False Positive
    FN_COLOR = (30, 144, 255)   # Blue - False Negative

    font = get_system_font(size=32)
    legend_font = get_system_font(size=50)

    gt_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        gt_by_image.setdefault(img_id, []).append(ann)

    def get_retinanet(num_classes):
        model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        return model
    
    model = get_retinanet(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform_tensor = transforms.Compose([transforms.ToTensor()])

    out_dir = os.path.join(output_dir, "GT_vs_Pred_correctness")
    os.makedirs(out_dir, exist_ok=True)

    processed = 0
    for img_info in coco_data["images"]:
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(test_img_dir, file_name)
        
        if not os.path.exists(img_path):
            continue
            
        img = Image.open(img_path).convert("RGB")

        gt_img = img.copy()
        gt_draw = ImageDraw.Draw(gt_img)

        gt_boxes = []
        gt_label_ids = []

        if img_id in gt_by_image:
            for ann in gt_by_image[img_id]:
                if ann["category_id"] not in preserved_ids:
                    continue
                x, y, w, h = ann["bbox"]
                x2, y2 = x + w, y + h

                cls_id = ann["category_id"]
                cls_name = class_names[cls_id]

                gt_boxes.append([x, y, x2, y2])
                gt_label_ids.append(cls_id)

                gt_draw.rectangle([x, y, x2, y2], outline=FN_COLOR, width=4)

                text = cls_name
                bbox = gt_draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                gt_draw.rectangle([x, y, x + tw + 12, y + th + 10], fill=FN_COLOR)
                gt_draw.text((x + 6, y + 5), text, fill="white", font=font)

        pred_img = img.copy()
        pred_draw = ImageDraw.Draw(pred_img)

        pred = model([transform_tensor(img).to(device)])[0]

        pred_boxes = pred["boxes"].cpu().tolist()
        pred_labels = pred["labels"].cpu().tolist()
        pred_scores = pred["scores"].cpu().tolist()

        detections = [
            (pb, pl, ps)
            for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores)
            if ps >= score_thresh
        ]

        matched_gt = set()

        for pb, pl, ps in detections:
            if pl not in preserved_ids:
                continue
            x1, y1, x2, y2 = pb
            cls_name = class_names[pl]

            best_iou = 0
            best_gt_idx = -1

            for idx, gb in enumerate(gt_boxes):
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_thresh:
                color = TP_COLOR  # Green - True Positive
                matched_gt.add(best_gt_idx)

                if pl in preserved_ids:
                    DETECTION_RESULTS.append((pl, pl))
            else:
                color = FP_COLOR  # Red - False Positive

                if pl in preserved_ids:
                    DETECTION_RESULTS.append((None, pl))

            pred_draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            text = f"{cls_name} {ps:.2f}"
            bbox = pred_draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            pred_draw.rectangle([x1, y1, x1 + tw + 12, y1 + th + 10], fill=color)
            pred_draw.text((x1 + 6, y1 + 5), text, fill="white", font=font)

        for i, gb in enumerate(gt_boxes):
            if gt_label_ids[i] not in preserved_ids:
                continue
            if i not in matched_gt:
                x1, y1, x2, y2 = gb
                pred_draw.rectangle([x1, y1, x2, y2], outline=FN_COLOR, width=3)

                if gt_label_ids[i] in preserved_ids:
                    DETECTION_RESULTS.append((gt_label_ids[i], None))

        w, h = img.size
        legend_height = 60

        combined = Image.new("RGB", (w * 2, h + legend_height), (230, 230, 230))

        legend = ImageDraw.Draw(combined)
        legend.rectangle([0, 0, w, legend_height], fill=(180, 220, 255))
        legend.rectangle([w, 0, w * 2, legend_height], fill=(255, 200, 170))

        legend.text((10, 8), "Ground Truth", fill="black", font=legend_font)
        legend.text((w + 10, 8), "Prediction", fill="black", font=legend_font)

        combined.paste(gt_img, (0, legend_height))
        combined.paste(pred_img, (w, legend_height))

        combined.save(os.path.join(out_dir, file_name))
        processed += 1
        
        if processed % 10 == 0:
            print(f"   Processed {processed} images...")

    print(f" Saved {processed} GT-vs-Pred images → {out_dir}")
    
    return DETECTION_RESULTS
