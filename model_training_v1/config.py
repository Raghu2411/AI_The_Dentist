"""
Configuration Management
Handles all configuration and path management for the training pipeline
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PathConfig:
    """Manage all files paths"""
    base_path: str
    output_dir: str = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = os.path.join(self.base_path, 'training_outputs')
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def train_json(self) -> str:
        return os.path.join(self.base_path, 'train/_annotations.coco.json')
    
    @property
    def val_json(self) -> str:
        return os.path.join(self.base_path, 'valid/_annotations.coco.json')
    
    @property
    def test_json(self) -> str:
        return os.path.join(self.base_path, 'test/_annotations.coco.json')
    
    @property
    def train_img_dir(self) -> str:
        return os.path.join(self.base_path, 'train')
    
    @property
    def val_img_dir(self) -> str:
        return os.path.join(self.base_path, 'valid')
    
    @property
    def test_img_dir(self) -> str:
        return os.path.join(self.base_path, 'test')
    
    def get_model_output_dir(self, model_name: str) -> str:
        """Get output directory for the model"""
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    num_epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-4
    patience: int = 15
    weight_decay: float = 1e-4
    
    pos_weight_multiplier: float = 1
    use_dynamic_pos_weights: bool = True
    _pos_weights: torch.Tensor = None
    
    def calculate_pos_weights(self, train_json: str, device: torch.device, num_classes: int = None) -> torch.Tensor:
        """
        Calculate positive class weights based on class imbalance in training data
        
        Args:
            train_json: Path to training COCO JSON file
            device: torch device to place weights on
            num_classes: Number of classes
            
        Returns:
            Tensor of positive weights for each class
        """
        import json
        
        with open(train_json, 'r') as f:
            data = json.load(f)
        
        category_ids = sorted({cat['id'] for cat in data['categories']})
        cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
        num_classes = num_classes or len(category_ids)
        
        class_counts = torch.zeros(num_classes)
        total_images = len(data['images'])
        
        for ann in data['annotations']:
            cat_id = ann['category_id']
            label_idx = cat_id_to_label[cat_id]
            class_counts[label_idx] += 1
        
        if self.use_dynamic_pos_weights:
            neg_counts = total_images - class_counts
            pos_weights = neg_counts / (class_counts + 1e-6)
            
            pos_weights = torch.clamp(pos_weights, min=1.0, max=50.0)
        else:
            multiplier = self.pos_weight_multiplier if self.pos_weight_multiplier is not None else 5.0
            pos_weights = torch.ones(num_classes) * multiplier
        
        pos_weights = pos_weights.to(device)
        
        self.pos_weight_multiplier = pos_weights
        
        print("\n" + "="*80)
        print("CLASS IMBALANCE ANALYSIS")
        print("="*80)
        print(f"Total images: {total_images}")
        print(f"Weight calculation: {'Dynamic (based on class distribution)' if self.use_dynamic_pos_weights else f'Fixed (multiplier={pos_weights[0].item()})'}")
        print(f"\nPer-class statistics:")
        print(f"{'Class':<30} {'Count':>10} {'Frequency':>12} {'Pos_Weight':>12}")
        print("-"*80)
        
        for i, cat_id in enumerate(category_ids):
            cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
            count = int(class_counts[i])
            freq = count / total_images if total_images > 0 else 0
            weight = pos_weights[i].item()
            print(f"{cat_name:<30} {count:>10} {freq:>12.2%} {weight:>12.2f}")
        
        print("-"*80)
        print(f"Mean pos_weight: {pos_weights.mean().item():.2f}")
        print(f"Median pos_weight: {pos_weights.median().item():.2f}")
        print(f"Min pos_weight: {pos_weights.min().item():.2f}")
        print(f"Max pos_weight: {pos_weights.max().item():.2f}")
        print("="*80 + "\n")
        
        return pos_weights
    
    def get_pos_weights(self, device: torch.device = None) -> torch.Tensor:
        """
        Get the calculated pos_weights
        
        Args:
            device: torch device (optional, will use stored device)
            
        Returns:
            Tensor of positive weights, or raises error if not calculated yet
        """
        if self.pos_weight_multiplier is None:
            raise ValueError("pos_weights not calculated yet. Call calculate_pos_weights() first.")
        
        if isinstance(self.pos_weight_multiplier, torch.Tensor):
            if device is not None:
                return self.pos_weight_multiplier.to(device)
            return self.pos_weight_multiplier
        else:
            raise ValueError("pos_weight_multiplier is a scalar. Call calculate_pos_weights() first.")


@dataclass
class ModelConfig:
    """Model specific configurations"""
    classification_models: List[str] = field(default_factory=lambda: [
        'vision_transformer',
        'resnet50'
    ])
    detection_models: List[str] = field(default_factory=lambda: [
        'yolov8n', 
        'yolov8m',
        'yolov8x',
        'yolov10n', 
        'yolov10m',
        'yolov10x',
        'retinanet'
        ])    
    
    yolo_img_size: int = 640
    yolo_batch_size: int = 2
    detection_score_threshold: float = 0.25
    detection_iou_threshold: float = 0.5


class Config:
    """Main configuration class that combines all configs"""
    
    def __init__(self, base_path: str):
        self.paths = PathConfig(base_path)
        self.training = TrainingConfig()
        self.models = ModelConfig()
        self.device = self._setup_device()
        self.preserved_classes = [
            'calculus', 'caries', 'crown', 'impacted', 'implant',
            'periapical radiolucency', 'rc-treated', 'restoration', 'root-stump'
        ]
    
    def _setup_device(self) -> torch.device:
        """Setup and configure device"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {gpu_memory:.2f} GB")
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        return device