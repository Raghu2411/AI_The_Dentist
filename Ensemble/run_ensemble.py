# Main Script

import os,sys
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import config
import model_utils
import post_processing
import visualization

original_sys_path = sys.path.copy()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'C:/Users/USER/Desktop/OPG-segmentation/Part 2/final/Dental_Dissertation')))
from train_other_models import DetectionOPGDataset
sys.path = original_sys_path

def main():
    print("Starting Ensemble Inference Pipeline...")
    print(f"Test Image Directory: {config.TEST_IMG_DIR}")
    print(f"Test Annotation JSON: {config.TEST_JSON_PATH}")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    
    print("\n--- Loading Models ---")
    yolo_model = model_utils.load_yolo(config.MODEL_PATHS['yolov9'])
    frcnn_model = model_utils.load_torchvision_model('faster_rcnn', config.MODEL_PATHS['faster_rcnn'], config.NUM_CLASSES)
    retina_model = model_utils.load_torchvision_model('retinanet', config.MODEL_PATHS['retinanet'], config.NUM_CLASSES)
    d2_predictor = model_utils.load_detectron2(config.CONFIG_PATHS['detectron2'], config.MODEL_PATHS['detectron2'])
    mmd_model = model_utils.load_mmdetection(config.CONFIG_PATHS['mmdetection'], config.MODEL_PATHS['mmdetection'])

    print("\n Loading Test Dataset")
    # Transforms will load images manually
    test_dataset = DetectionOPGDataset(config.TEST_JSON_PATH, config.TEST_IMG_DIR, transform=None)
    
    final_json_output = []

    print(f"\n Starting Inference on {len(test_dataset)} Test Images ")
    for i in tqdm(range(len(test_dataset))):
        img_id = test_dataset.image_ids_with_ann[i]
        img_info = test_dataset.images_map[img_id]
        file_name = img_info['file_name']
        image_path = os.path.join(config.TEST_IMG_DIR, file_name)
        
        try:
            image_pil = Image.open(image_path).convert('RGB')
            image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Warning: Skipping image {file_name}. Error: {e}")
            continue

        # Run Inference 
        all_predictions = []
        all_predictions.extend(model_utils.run_yolo_inference(yolo_model, image_cv2))
        all_predictions.extend(model_utils.run_detectron2_inference(d2_predictor, image_cv2))
        # all_predictions.extend(model_utils.run_mmdetection_inference(mmd_model, image_cv2))
        # all_predictions.extend(model_utils.run_torchvision_inference(frcnn_model, image_pil, 'faster_rcnn'))
        # all_predictions.extend(model_utils.run_torchvision_inference(retina_model, image_pil, 'retinanet'))

        # Post-processing 
        
        # Filter by confidence
        filtered_preds = post_processing.filter_by_confidence(all_predictions)
        
        # NMS (Class-aware)
        final_preds_list = post_processing.apply_batched_nms(filtered_preds, config.NMS_IOU_THRESHOLD)
        
        # Assign tooth-related observations
        structured_teeth, unassigned_diseases = post_processing.assign_diseases_to_teeth(final_preds_list)
        
        # Insert domain knowledge rules
        structured_teeth = post_processing.apply_domain_rules(structured_teeth)

        final_structured_predictions = {
            "teeth": structured_teeth,
            "unassigned": unassigned_diseases
        }
        
        # Visualization 
        gt_annotations = test_dataset.annotations_by_image[img_id]
        output_image_path = os.path.join(config.OUTPUT_DIR, file_name)
        visualization.create_comparison_image(
            image_pil,
            gt_annotations,
            final_structured_predictions,
            output_image_path
        )
        
        # Build Spatial Relationships List
        spatial_relationships = []
        for tooth in structured_teeth:
            for condition in tooth['conditions']:
                spatial_relationships.append({
                    "tooth_id": tooth['class_name'],
                    "disease": condition['class_name'],
                    "disease_bbox": condition['bbox'],
                    "disease_model": condition['model_source']
                })
        
        # Save JSON Output 
        final_json_output.append({
            "image_id": img_id,
            "image_name": file_name,
            "predictions": final_structured_predictions,
            "spatial_relationships": spatial_relationships
        })

    output_json_path = os.path.join(config.OUTPUT_DIR, '_ensemble_annotations.json')
    with open(output_json_path, 'w') as f:
        json.dump(final_json_output, f, indent=4)
        
    print(f"\n--- Pipeline Complete ---")
    print(f"Ensemble predictions saved to: {config.OUTPUT_DIR}")
    print(f"Ensemble JSON saved to: {output_json_path}")

if __name__ == "__main__":
    main()
