
"""
Training Strategies
Implements different training strategies
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


class TrainingStrategy(ABC):
    """Abstract base class for training strategies"""
    
    def __init__(self, config, model_name: str):
        self.config = config
        self.model_name = model_name
        self.device = config.device
        self.output_dir = config.paths.get_model_output_dir(
            self._get_output_folder_name()
        )
    
    @abstractmethod
    def _get_output_folder_name(self) -> str:
        """Get the output folder name for this strategy"""
        pass
    
    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """Train the model"""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate the model"""
        pass
    
    def save_model(self, model: nn.Module, filename: str):
        """Save model weights"""
        save_path = os.path.join(self.output_dir, filename)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved: {save_path}")
        return save_path
    
    def load_model(self, model: nn.Module, filepath: str):
        """Load model weights"""
        model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded: {filepath}")
        return model
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class ClassificationTrainingStrategy(TrainingStrategy):
    """Training strategy for classification models"""
    
    def _get_output_folder_name(self) -> str:
        return f"classification_{self.model_name}"
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        num_classes: int,
        pos_weight: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ) -> nn.Module:
        """
        Train a classification model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model: Model to train
            num_classes: Number of classes
            pos_weight: Positive class weights
            class_names: List of class names
            
        Returns:
            Trained model
        """
        from evaluators import ClassificationEvaluator
        from visualizers import LearningCurveVisualizer
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        best_f1 = -1
        patience_counter = 0
        history = []
        
        evaluator = ClassificationEvaluator(self.device)
        
        for epoch in range(self.config.training.num_epochs):
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, criterion
            )
            
            val_metrics = evaluator.evaluate(
                model, val_loader, criterion, class_names
            )
            
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs}, "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Train F1: {train_metrics['f1_weighted']:.4f}, "
                  f"Val F1: {val_metrics['f1_weighted']:.4f}")
            
            history.append({
                "Epoch": epoch + 1,
                "Train_Loss": train_metrics['loss'],
                "Val_Loss": val_metrics['loss'],
                "Train_F1_Weighted": train_metrics['f1_weighted'],
                "Val_F1_Weighted": val_metrics['f1_weighted']
            })
            
            if val_metrics['f1_weighted'] > best_f1:
                best_f1 = val_metrics['f1_weighted']
                self.save_model(model, f'best_model_{self.model_name}.pth')
                patience_counter = 0
                print(f"New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.patience:
                    print("Early stopping triggered")
                    break
        
        visualizer = LearningCurveVisualizer(self.output_dir)
        visualizer.plot_classification_curves(history, self.model_name)
        
        return model
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Train for one epoch"""
        
        model.train()
        running_loss = 0
        all_preds, all_labels = [], []
        
        for images, labels, _ in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).long().cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        f1_weighted = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )[2]
        
        return {
            'loss': running_loss / len(loader),
            'f1_weighted': f1_weighted
        }
    
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: List[str],
        preserved_classes: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate classification model"""
        from evaluators import ClassificationEvaluator
        
        evaluator = ClassificationEvaluator(self.device)
        criterion = nn.BCEWithLogitsLoss()
        
        results = evaluator.evaluate(
            model, test_loader, criterion, class_names, preserved_classes
        )
        
        print("\n" + "="*80)
        print("=== OVERALL METRICS (ALL CLASSES) ===")
        print("="*80)
        print(f"Loss: {results['loss']:.4f}")
        print(f"Subset Accuracy: {results['subset_accuracy']:.4f}")
        print(f"Accuracy (Macro): {results['accuracy_macro']:.4f}")
        print(f"Precision (Macro): {results['precision_macro']:.4f}")
        print(f"Recall (Macro): {results['recall_macro']:.4f}")
        print(f"F1 (Macro): {results['f1_macro']:.4f}")
        print(f"Precision (Weighted): {results['precision_weighted']:.4f}")
        print(f"Recall (Weighted): {results['recall_weighted']:.4f}")
        print(f"F1 (Weighted): {results['f1_weighted']:.4f}")
        print(f"ROC-AUC (Macro): {results['roc_auc_macro']:.4f}")
        
        if results.get('preserved_metrics'):
            print("\n" + "="*80)
            print("=== PRESERVED CLASSES METRICS ===")
            print("="*80)
            pres = results['preserved_metrics']
            print(f"Subset Accuracy: {pres['subset_accuracy']:.4f}")
            print(f"Accuracy (Macro): {pres['accuracy_macro']:.4f}")
            print(f"Precision (Macro): {pres['precision_macro']:.4f}")
            print(f"Recall (Macro): {pres['recall_macro']:.4f}")
            print(f"F1 (Macro): {pres['f1_macro']:.4f}")
            print(f"Precision (Weighted): {pres['precision_weighted']:.4f}")
            print(f"Recall (Weighted): {pres['recall_weighted']:.4f}")
            print(f"F1 (Weighted): {pres['f1_weighted']:.4f}")
            print(f"ROC-AUC (Macro): {pres['roc_auc_macro']:.4f}")
        
        print("\n" + "="*80)
        print("=== PER-CLASS METRICS ===")
        print("="*80)
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"\n{class_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}" if isinstance(value, float) else f"  {metric_name}: {value}")
        
        return results


class DetectionTrainingStrategy(TrainingStrategy):
    """Training strategy for detection models"""
    
    def _get_output_folder_name(self) -> str:
        return f"detection_{self.model_name}"
    
    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """Train detection model to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate detection model to be implemented by subclasses"""
        pass