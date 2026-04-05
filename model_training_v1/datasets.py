"""
Dataset Management
Handles all dataset loading and preprocessing
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from typing import Tuple, Dict, List, Optional
from abc import ABC, abstractmethod


class BaseDataset(ABC, Dataset):
    """Abstract base class for all datasets"""
    
    def __init__(self, json_file: str, img_dir: str, transform=None):
        self.json_file = json_file
        self.img_dir = img_dir
        self.transform = transform
        self._load_annotations()
    
    @abstractmethod
    def _load_annotations(self):
        """Load and process annotations"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        """Get dataset item"""
        pass
    
    @abstractmethod
    def __len__(self):
        """Get dataset length"""
        pass


class ClassificationDataset(BaseDataset):
    """Dataset for multi label classification"""
    
    def _load_annotations(self):
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        category_ids = sorted({cat['id'] for cat in self.data['categories']})
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
        cat_id_to_name = {cat['id']: cat['name'] for cat in self.data['categories']}
        self.categories = {idx: cat_id_to_name[cat_id] for cat_id, idx in self.cat_id_to_label.items()}
        
        self.images_with_annotations = [
            img for img in self.data['images']
            if any(ann['image_id'] == img['id'] for ann in self.data['annotations'])
        ]
        
        self.num_classes = len(self.categories)
        self.class_names = [self.categories[i] for i in range(self.num_classes)]
    
    def __len__(self) -> int:
        return len(self.images_with_annotations)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_info = self.images_with_annotations[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        
        annotations = [ann for ann in self.data['annotations'] if ann['image_id'] == img_info['id']]
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        
        for ann in annotations:
            category_id = ann['category_id']
            label_idx = self.cat_id_to_label[category_id]
            label[label_idx] = 1.0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_info['file_name']


class DetectionDataset(BaseDataset):
    """Dataset for detection"""
    
    def _load_annotations(self):
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        self.categories = sorted(self.data['categories'], key=lambda x: x['id'])
        self.cat_id_to_label = {c['id']: i for i, c in enumerate(self.categories)}
        self.class_names = [c['name'] for c in self.categories]
        
        self.annotations_by_image = {img['id']: [] for img in self.data['images']}
        for ann in self.data['annotations']:
            self.annotations_by_image[ann['image_id']].append(ann)
        
        self.image_ids_with_ann = [
            img_id for img_id, anns in self.annotations_by_image.items() if anns
        ]
        self.images_map = {img['id']: img for img in self.data['images']}
        
        print(f"Loaded {os.path.basename(self.json_file)}: "
              f"{len(self.image_ids_with_ann)} images, {len(self.class_names)} classes.")
    
    def __len__(self) -> int:
        return len(self.image_ids_with_ann)
    
    def __getitem__(self, idx) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        img_id = self.image_ids_with_ann[idx]
        img_info = self.images_map[img_id]
        
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        boxes, labels = [], []
        for ann in self.annotations_by_image[img_id]:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(self.cat_id_to_label[ann['category_id']])
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def collate_fn(batch):
    """Custom collate function for detection datasets"""
    return tuple(zip(*batch))


class DatasetFactory:
    """Factory for creating datasets"""
    
    @staticmethod
    def create_classification_dataset(json_file: str, img_dir: str, transform=None) -> ClassificationDataset:
        """Create classification dataset"""
        return ClassificationDataset(json_file, img_dir, transform)
    
    @staticmethod
    def create_detection_dataset(json_file: str, img_dir: str, transform=None) -> DetectionDataset:
        """Create detection dataset"""
        return DetectionDataset(json_file, img_dir, transform)
