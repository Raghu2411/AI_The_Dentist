"""
Model Factory
Creates and configures different model architectures
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from ultralytics import YOLO
from typing import Tuple
from abc import ABC, abstractmethod


class ModelBuilder(ABC):
    """Abstract base class for model builders"""
    
    @abstractmethod
    def build(self, num_classes: int, device: torch.device) -> nn.Module:
        """Build and return the model"""
        pass


class VisionTransformerBuilder(ModelBuilder):
    """Builder for Vision Transformer model"""
    
    def build(self, num_classes: int, device: torch.device) -> nn.Module:
        model = models.vit_b_16(weights='DEFAULT')
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model.to(device)


class ResNet50Builder(ModelBuilder):
    """Builder for ResNet50 model"""
    
    def build(self, num_classes: int, device: torch.device) -> nn.Module:
        model = models.resnet50(weights='DEFAULT')
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model.to(device)


class RetinaNetBuilder(ModelBuilder):
    """Builder for RetinaNet model"""
    
    def build(self, num_classes: int, device: torch.device) -> nn.Module:
        model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        return model.to(device)


class YOLOBuilder(ModelBuilder):
    """Builder for YOLO models"""
    
    def __init__(self, model_variant: str):
        self.model_variant = model_variant
    
    def build(self, num_classes: int = None, device: torch.device = None) -> YOLO:
        """
        YOLO models are built differently they need a .pt file path
        num_classes and device are handled during training
        """
        return YOLO(f'{self.model_variant}.pt')


class ModelFactory:
    """Factory for creating different model types"""
    
    _builders = {
        'vision_transformer': VisionTransformerBuilder,
        'resnet50': ResNet50Builder,
        'retinanet': RetinaNetBuilder,
    }
    
    @classmethod
    def create_model(cls, model_name: str, num_classes: int, device: torch.device) -> nn.Module:
        """
        Create a model by name
        
        Args:
            model_name: Name of the model
            num_classes: Number of output classes
            device: Device to place model on
            
        Returns:
            Configured model
        """
        if model_name.startswith('yolo'):
            builder = YOLOBuilder(model_name)
            return builder.build()
        
        builder_class = cls._builders.get(model_name)
        if builder_class is None:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available models: {list(cls._builders.keys())}")
        
        builder = builder_class()
        return builder.build(num_classes, device)
    
    @classmethod
    def register_builder(cls, name: str, builder_class: type):
        """Register a new model builder"""
        cls._builders[name] = builder_class
    
    @classmethod
    def get_model_type(cls, model_name: str) -> str:
        """Get the type of model"""
        if model_name.startswith('yolo') or model_name == 'retinanet':
            return 'detection'
        return 'classification'
    
    @classmethod
    def get_gradcam_layer(cls, model_name: str, model: nn.Module):
        """Get the target layer for GradCAM visualization"""
        layer_map = {
            'resnet50': lambda m: m.layer4[-1],
            'vision_transformer': lambda m: m.encoder.layers[-1].ln_1,
        }
        
        if model_name not in layer_map:
            raise ValueError(f"GradCAM not supported for {model_name}")
        
        return layer_map[model_name](model)
