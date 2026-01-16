import json
import os
import shutil

def convert_coco_to_yolo(coco_json_path, image_source_dir, yolo_base_dir, split):
    """
    Converts a COCO annotation file and its corresponding images to YOLO format.

    Args:
        coco_json_path (str): Path to the COCO JSON annotation file.
        image_source_dir (str): Directory where the original images are stored.
        yolo_base_dir (str): The base directory to save the YOLO formatted data.
        split (str): The dataset split, e.g., 'train', 'valid', or 'test'.
    """
    print(f"--- Processing {split} set ---")
    
    yolo_images_dir = os.path.join(yolo_base_dir, 'images', split)
    yolo_labels_dir = os.path.join(yolo_base_dir, 'labels', split)
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    # Load COCO data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
        
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # Create a mapping from COCO category_id to YOLO class_index (0-indexed)
    category_id_to_class_index = {cat['id']: i for i, cat in enumerate(categories)}

    annotations_by_image = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
        
    # Convert and write YOLO annotation files
    for image_id, image_info in images.items():
        if image_id not in annotations_by_image:
            continue # Skip images with no annotations
            
        img_width = image_info['width']
        img_height = image_info['height']
        file_name = image_info['file_name']
        
        label_file_name = os.path.splitext(file_name)[0] + '.txt'
        label_file_path = os.path.join(yolo_labels_dir, label_file_name)
        
        with open(label_file_path, 'w') as f:
            for ann in annotations_by_image[image_id]:
                category_id = ann['category_id']
                class_index = category_id_to_class_index[category_id]
                
                # COCO bbox is [x_min, y_min, width, height]
                x_min, y_min, box_width, box_height = ann['bbox']
                
                # Convert to YOLO format (normalized center_x, center_y, width, height)
                x_center = (x_min + box_width / 2) / img_width
                y_center = (y_min + box_height / 2) / img_height
                norm_width = box_width / img_width
                norm_height = box_height / img_height
                
                f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                
        shutil.copy(os.path.join(image_source_dir, file_name), os.path.join(yolo_images_dir, file_name))
        
    print(f"Successfully converted {len(images)} images and their annotations for the {split} set.")
    return categories

def create_data_yaml(yolo_base_dir, categories):
    """
    Creates the data.yaml file required by YOLO for training.
    """
    class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    num_classes = len(class_names)
    
    yolo_base_dir_abs = os.path.abspath(yolo_base_dir)
    
    yaml_content = f"""
train: {os.path.join(yolo_base_dir_abs, 'images/train')}
val: {os.path.join(yolo_base_dir_abs, 'images/valid')}
test: {os.path.join(yolo_base_dir_abs, 'images/test')}

# number of classes
nc: {num_classes}

# class names
names: {class_names}
"""
    yaml_path = os.path.join(yolo_base_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nSuccessfully created 'data.yaml' at {yaml_path}")
    

if __name__ == '__main__':
    coco_base_dir = '/home/tq24401/llm/filtered_opg_dataset_for_LLM'
    yolo_output_dir = '/home/tq24401/llm/YOLO'
    datasets = {
        'train': os.path.join(coco_base_dir, 'train'),
        'valid': os.path.join(coco_base_dir, 'valid'),
        'test': os.path.join(coco_base_dir, 'test')
    }
    
    final_categories = None
    for split, path in datasets.items():
        json_file = os.path.join(path, '_annotations.coco.json')
        image_dir = path
        categories = convert_coco_to_yolo(json_file, image_dir, yolo_output_dir, split)
        if final_categories is None:
            final_categories = categories

    if final_categories:
        create_data_yaml(yolo_output_dir, final_categories)
    
    print("\nConversion complete!")