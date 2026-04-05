"""
Model Trainer
Main class that coordinates the entire training pipeline
"""

import torch
from torch.utils.data import DataLoader
import os
import warnings

from config import Config
from datasets import DatasetFactory, collate_fn
from model_factory import ModelFactory
from transforms import TransformFactory
from training_strategies import ClassificationTrainingStrategy
from detection_trainers import RetinaNetTrainingStrategy, YOLOTrainingStrategy
from utils import COCOToYOLOConverter, CategoryLoader, clear_memory
# from gradcam import generate_gradcam_for_model

from visualize_gt_vs_pred_retinanet import visualize_gt_vs_pred_retinanet

warnings.filterwarnings("ignore", category=UserWarning)

class ModelTrainer:
    """ModelTrainer the entire training pipeline"""
    
    def __init__(self, base_path: str):
        """
        Initialize the training
        
        Args:
            base_path: Base directory containing train/valid/test splits
        """
        self.config = Config(base_path)
        self.categories = CategoryLoader.load_categories(self.config.paths.train_json)
        
        print("\n" + "="*80)
        print("DENTAL MODEL TRAINING PIPELINE")
        print("Classification: Vision Transformer, ResNet50")
        print("Object Detection: YOLOv8, YOLOv10, RetinaNet")
        print("="*80 + "\n")
        print(f"Classes ({len(self.categories)}): {self.categories}\n")
    
    
    def train_all_models(self):
        """Train all configured models"""
        models_to_train = (
            self.config.models.classification_models +
            self.config.models.detection_models
        )
        
        for model_name in models_to_train:
            print(f"\n{'='*80}")
            print(f"MODEL: {model_name.upper()}")
            print(f"{'='*80}\n")
            
            clear_memory()
            
            model_type = ModelFactory.get_model_type(model_name)
            
            if model_type == 'classification':
                self._train_classification_model(model_name)
            elif model_type == 'detection':
                self._train_detection_model(model_name)
        
        print("\n" + "="*80)
        print("ALL TRAINING COMPLETE!")
        print("="*80)
    
    def _train_classification_model(self, model_name: str):
        """Train a classification model"""
        print(f"\n--- Training Classification Model: {model_name} ---")
        
        transform = TransformFactory.get_classification_transforms(model_name)
        
        train_dataset = DatasetFactory.create_classification_dataset(
            self.config.paths.train_json,
            self.config.paths.train_img_dir,
            transform
        )
        val_dataset = DatasetFactory.create_classification_dataset(
            self.config.paths.val_json,
            self.config.paths.val_img_dir,
            transform
        )
        test_dataset = DatasetFactory.create_classification_dataset(
            self.config.paths.test_json,
            self.config.paths.test_img_dir,
            transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        
        model = ModelFactory.create_model(
            model_name,
            train_dataset.num_classes,
            self.config.device
        )
        
        pos_weight = torch.ones(train_dataset.num_classes).to(self.config.device)
        self.config.training.calculate_pos_weights(self.config.paths.train_json, pos_weight)
        pos_weight *= self.config.training.pos_weight_multiplier
        
        print(f"Pos Weights: {pos_weight.cpu().numpy()}\n")
        strategy = ClassificationTrainingStrategy(self.config, model_name)
        print("\n--- Starting Training ---")
        model = strategy.train(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            num_classes=train_dataset.num_classes,
            pos_weight=pos_weight,
            class_names=train_dataset.class_names
        )
        
        model_path = os.path.join(
            strategy.output_dir,
            f'best_model_{model_name}.pth'
        )
        strategy.load_model(model, model_path)
        
        print("\n--- Evaluating on Test Set ---")
        strategy.evaluate(
            model=model,
            test_loader=test_loader,
            class_names=train_dataset.class_names,
            preserved_classes=self.config.preserved_classes,
        )
        
        strategy.cleanup_memory()
    
    def _train_detection_model(self, model_name: str):
        """Train a detection model"""
        if model_name.startswith('yolo'):
            self._train_yolo_model(model_name)
        elif model_name == 'retinanet':
            self._train_retinanet_model(model_name)
    
    def _train_yolo_model(self, model_name: str):
        """Train a YOLO model"""
        converter = COCOToYOLOConverter(self.config.paths.base_path)
        
        if not converter.check_dataset_exists():
            print("Converting COCO to YOLO format...")
            converter.convert()
        else:
            print("YOLO dataset exists\n")
        
        data_yaml = converter.create_data_yaml(self.categories)
        
        model = ModelFactory.create_model(model_name, None, None)
        
        strategy = YOLOTrainingStrategy(self.config, model_name)
        model_path = strategy.train(
            data_yaml=data_yaml,
            model=model
        )
        
        if model_path:
            strategy.evaluate(
                model_path=model_path,
                data_yaml=data_yaml,
                class_names=self.categories,
                preserved_classes=self.config.preserved_classes
            )
        
        strategy.cleanup_memory()
    
    def _train_retinanet_model(self, model_name: str):
        """Train a RetinaNet model"""
        train_transform = TransformFactory.get_detection_train_transforms()
        val_transform = TransformFactory.get_detection_val_transforms()
        
        train_dataset = DatasetFactory.create_detection_dataset(
            self.config.paths.train_json,
            os.path.join(self.config.paths.train_img_dir, 'images'),
            train_transform
        )
        val_dataset = DatasetFactory.create_detection_dataset(
            self.config.paths.val_json,
            os.path.join(self.config.paths.val_img_dir, 'images'),
            val_transform
        )
        test_dataset = DatasetFactory.create_detection_dataset(
            self.config.paths.test_json,
            os.path.join(self.config.paths.test_img_dir, 'images'),
            val_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )
        
        model = ModelFactory.create_model(
            model_name,
            len(self.categories),
            self.config.device
        )
        
        strategy = RetinaNetTrainingStrategy(self.config, model_name)
        model = strategy.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model
        )
        
        model_path = os.path.join(strategy.output_dir, "final_retinanet.pth")
        strategy.load_model(model, model_path)
        
        print("\n--- Final Test Evaluation ---")
        strategy.evaluate(
            model=model,
            test_loader=test_loader,
            class_names=self.categories,
            preserved_classes=self.config.preserved_classes
        )
        
        print("\n--- Generating RetinaNet Visualizations ---")

        visualize_gt_vs_pred_retinanet(
            model_path=model_path,
            base_path=self.config.paths.base_path,
            device=self.config.device,
            output_dir=strategy.output_dir,
            score_thresh=0.25,
            iou_thresh=0.5,
            preserved_classes=self.config.preserved_classes
        )
        
        strategy.cleanup_memory()

def main():
    """Main entry point"""
    base_path = '/home/mb24134/Dissertation'
    # base_path = '/Users/muhammadbabar/essex-work/FYPLatest/mobile-app/essex_dental_model_training'
    
    orchestrator = ModelTrainer(base_path)
    orchestrator.train_all_models()

if __name__ == "__main__":
    main()


4

1- 2nd image mobile dection
2- mobile chatbot image 
3- Hugging face chatbot
4- Feedback form image
