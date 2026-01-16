# Step 6 + 7, 8

import torch
import numpy as np
import torchvision.ops as ops
import config

def calculate_iou(box1, box2):
    """Calculates IoU between two boxes [x1, y1, x2, y2]."""
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
    """
    Filters a list of standard predictions using the class-specific thresholds.
    Can handle single minimum thresholds (float) or ranges (list/tuple).
    """
    filtered_preds = []
    for pred in all_predictions:
        class_name = pred['class_name']
        threshold_value = config.CLASS_THRESHOLDS.get(class_name, 0.3)
        
        if isinstance(threshold_value, (list, tuple)) and len(threshold_value) == 2:
            # It's a range: [min, max]
            min_thresh = threshold_value[0]
            max_thresh = threshold_value[1]
            if min_thresh <= pred['score'] <= max_thresh:
                filtered_preds.append(pred)
        elif isinstance(threshold_value, (float, int)):
            # A single threshold
            if pred['score'] >= threshold_value:
                filtered_preds.append(pred)
        else:
            if pred['score'] >= 0.3:
                filtered_preds.append(pred)
                
    return filtered_preds

def apply_batched_nms(predictions, iou_threshold):
    """
    Applies class-aware NMS to the list of predictions.
    This keeps the "best" prediction (highest score) for any overlapping
    boxes of the same class, and automatically keeps its `model_source`.
    """
    if not predictions:
        return []

    boxes = torch.tensor([p['bbox'] for p in predictions], dtype=torch.float32)
    scores = torch.tensor([p['score'] for p in predictions], dtype=torch.float32)
    class_idxs = torch.tensor([p['class_id'] for p in predictions], dtype=torch.long)
    
    keep_indices = ops.batched_nms(boxes, scores, class_idxs, iou_threshold)
    
    final_predictions = [predictions[i] for i in keep_indices]
    return final_predictions

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
                best_teeth[class_name] = tooth
                
    return list(best_teeth.values())

def assign_diseases_to_teeth(final_predictions):
    teeth = []
    diseases = []
    unassigned_diseases = []

    for pred in final_predictions:
        if pred['class_name'] in config.FDI_CLASSES:
            teeth.append(pred)
        elif pred['class_name'] == 'implant': # Not the Implant
            unassigned_diseases.append(pred)
        else:
            diseases.append(pred)
    
    # Apply deduplication ONLY to the FDI tooth list
    teeth = deduplicate_teeth_by_score(teeth)
    
    structured_teeth = []
    for tooth in teeth:
        tooth_dict = {
            **tooth, 
            "conditions": [],
            "recommendation": None,
            "recommendation_reason": None
        }
        
        for i, disease in enumerate(diseases):
            if disease.get('assigned', False):
                continue
            iou = calculate_iou(tooth['bbox'], disease['bbox'])
            if iou >= config.ASSIGNMENT_IOU_THRESHOLD:
                tooth_dict['conditions'].append(disease)
                diseases[i]['assigned'] = True
        structured_teeth.append(tooth_dict)

    # Add any diseases that were not assigned to any tooth
    unassigned = [d for d in diseases if not d.get('assigned', False)]
    # Add the implants back to the unassigned list
    unassigned.extend(unassigned_diseases)
    
    return structured_teeth, unassigned

def apply_domain_rules(structured_teeth):
    """
    Apply domain knowledge rules.
    """
    present_teeth = {t['class_name'] for t in structured_teeth}
    
    for tooth in structured_teeth:
        # Recommend extraction for unopposed wisdom teeth
        if tooth['class_name'] == '18' or '48' not in present_teeth:
            tooth['recommendation'] = 'Extraction'
            tooth['recommendation_reason'] = 'Unopposed opposing wisdom tooth'
        
        if tooth['class_name'] == '28' or '38' not in present_teeth:
            tooth['recommendation'] = 'Extraction'
            tooth['recommendation_reason'] = 'Unopposed opposing wisdom tooth'
        
        # Recommendation: for severe caries
        # for cond in tooth['conditions']:
        #     if cond['class_name'] == 'caries' and cond['score'] > 0.8:
        #         tooth['recommendation'] = 'Checkup'
        #         tooth['recommendation_reason'] = 'High confidence caries detected'
            
    return structured_teeth
