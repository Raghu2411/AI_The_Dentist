"""
Transform Factory
Creates appropriate transforms for models
"""

from torchvision import transforms
from typing import Optional


class TransformFactory:
    """Factory for creating data transforms"""
    
    @staticmethod
    def get_classification_transforms(model_name: str) -> transforms.Compose:
        """
        Get transforms for classification models
        
        Args:
            model_name: Name of the model
            
        Returns:
            Composed transforms
        """
        if model_name in ['resnet50', 'vision_transformer']:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            raise ValueError(f"Unknown model for transforms: {model_name}")
    
    @staticmethod
    def get_detection_train_transforms() -> transforms.Compose:
        """Get transforms for detection training"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    
    @staticmethod
    def get_detection_val_transforms() -> transforms.Compose:
        """Get transforms for detection validation/testing"""
        return transforms.Compose([
            transforms.ToTensor()
        ])
    
    @staticmethod
    def create_augmented_transforms(
        model_name: str,
        rotation_range: Optional[int] = None,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None
    ) -> transforms.Compose:
        """
        Create augmented transforms with custom parameters
        
        Args:
            model_name: Name of the model
            rotation_range: Rotation range in degrees
            brightness: Brightness adjustment factor
            contrast: Contrast adjustment factor
            
        Returns:
            Composed transforms with augmentation
        """
        transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        
        if rotation_range:
            transform_list.append(
                transforms.RandomRotation(rotation_range)
            )
        
        if brightness or contrast:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness or 0,
                    contrast=contrast or 0
                )
            )
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return transforms.Compose(transform_list)
