"""
Utility Functions
Helper functions for data conversion, visualization
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple
import yaml
import gc
import torch

class COCOToYOLOConverter:
    """Converts COCO format annotations to YOLO format"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def convert(self, splits: List[str] = ['train', 'valid', 'test']):
        """Convert COCO annotations to YOLO format"""
        print("\n" + "="*80)
        print("CONVERTING COCO TO YOLO FORMAT")
        print("="*80)
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            self._convert_split(split)
        
        print("\nCONVERSION COMPLETE\n")
    
    def _convert_split(self, split: str):
        """Convert a single split"""
        coco_json = os.path.join(self.base_path, split, '_annotations.coco.json')
        images_src = os.path.join(self.base_path, split)
        images_dst = os.path.join(self.base_path, split, 'images')
        labels_dst = os.path.join(self.base_path, split, 'labels')
        
        os.makedirs(images_dst, exist_ok=True)
        os.makedirs(labels_dst, exist_ok=True)
        
        if not os.path.exists(coco_json):
            print(f"  ✗ Annotations not found: {coco_json}")
            return
        
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
        
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
        
        print(f"  Categories ({len(categories)}): {[cat['name'] for cat in categories]}")
        
        img_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)
        
        img_id_to_info = {img['id']: img for img in coco_data['images']}
        
        converted_count = 0
        for img_id, img_info in img_id_to_info.items():
            filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            src_img_path = os.path.join(images_src, filename)
            
            if not os.path.exists(src_img_path):
                continue
            
            dst_img_path = os.path.join(images_dst, filename)
            label_filename = Path(filename).stem + '.txt'
            label_path = os.path.join(labels_dst, label_filename)
            
            if not os.path.exists(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            
            if img_id in img_id_to_anns:
                with open(label_path, 'w') as f:
                    for ann in img_id_to_anns[img_id]:
                        class_idx = cat_id_to_idx[ann['category_id']]
                        bbox_coco = ann['bbox']
                        bbox_yolo = self._convert_bbox(
                            bbox_coco, img_width, img_height
                        )
                        f.write(f"{class_idx} {' '.join(map(str, bbox_yolo))}\n")
                converted_count += 1
            else:
                open(label_path, 'w').close()
        
        print(f" Converted {converted_count} images")
    
    @staticmethod
    def _convert_bbox(
        bbox: List[float],
        img_width: int,
        img_height: int
    ) -> List[float]:
        """Convert COCO bbox to YOLO format (normalized)"""
        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        width = width / img_width
        height = height / img_height
        return [x_center, y_center, width, height]
    
    def check_dataset_exists(self) -> bool:
        """Check if YOLO dataset structure exists"""
        required = [
            'train/images', 'train/labels',
            'valid/images', 'valid/labels',
            'test/images', 'test/labels'
        ]
        
        for req in required:
            if not os.path.exists(os.path.join(self.base_path, req)):
                return False
        return True
    
    def create_data_yaml(self, categories: List[str]) -> str:
        """Create data.yaml for YOLO training"""
        data_yaml_path = os.path.join(self.base_path, 'data.yaml')
        
        yaml_content = {
            'path': self.base_path,
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(categories),
            'names': categories
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f" Created data.yaml")
        return data_yaml_path


class CategoryLoader:
    """Loads and manages category information"""
    
    @staticmethod
    def load_categories(json_file: str) -> List[str]:
        """Load category names from COCO JSON"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        categories = [
            cat['name'] 
            for cat in sorted(data['categories'], key=lambda x: x['id'])
        ]
        
        return categories


def clear_memory():
    """Clear GPU and system memory"""
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
