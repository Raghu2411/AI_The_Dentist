"""
Model Evaluators
Handles evaluation metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score
)
from torchmetrics.detection import MeanAveragePrecision
from typing import Dict, Any, List
import numpy as np


class ClassificationEvaluator:
    """Evaluator for classification models"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        class_names: List[str],
        preserved_classes: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate classification model with comprehensive metrics
        
        Args:
            model: Model to evaluate
            loader: Data loader
            criterion: Loss criterion
            class_names: List of class names
            preserved_classes: Optional list of preserved classes for filtered metrics
            
        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).long()
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        loss = total_loss / len(loader)
        subset_accuracy = accuracy_score(all_targets, all_preds)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        
        precision_macro = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )[0]
        
        recall_macro = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )[1]
        
        f1_macro = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )[2]
        
        precision_weighted = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )[0]
        
        recall_weighted = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )[1]
        
        f1_weighted = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )[2]
        
        try:
            roc_auc_macro = roc_auc_score(all_targets, all_preds, average='macro')
        except ValueError:
            roc_auc_macro = float('nan')
        
        accuracy_macro = np.mean([
            accuracy_score(all_targets[:, i], all_preds[:, i])
            for i in range(all_targets.shape[1])
        ])
        
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            acc = accuracy_score(all_targets[:, i], all_preds[:, i])
            
            try:
                auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
            except ValueError:
                auc = float('nan')
            
            per_class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'accuracy': acc,
                'roc_auc': auc,
                'support': int(support[i])
            }
        
        preserved_metrics = None
        if preserved_classes is not None:
            preserved_indices = [
                i for i, class_name in enumerate(class_names) 
                if class_name in preserved_classes
            ]
            
            if preserved_indices:
                preserved_preds = all_preds[:, preserved_indices]
                preserved_targets = all_targets[:, preserved_indices]
                
                preserved_subset_accuracy = accuracy_score(preserved_targets, preserved_preds)
                
                preserved_precision_macro = precision_recall_fscore_support(
                    preserved_targets, preserved_preds, average='macro', zero_division=0
                )[0]
                
                preserved_recall_macro = precision_recall_fscore_support(
                    preserved_targets, preserved_preds, average='macro', zero_division=0
                )[1]
                
                preserved_f1_macro = precision_recall_fscore_support(
                    preserved_targets, preserved_preds, average='macro', zero_division=0
                )[2]
                
                preserved_precision_weighted = precision_recall_fscore_support(
                    preserved_targets, preserved_preds, average='weighted', zero_division=0
                )[0]
                
                preserved_recall_weighted = precision_recall_fscore_support(
                    preserved_targets, preserved_preds, average='weighted', zero_division=0
                )[1]
                
                preserved_f1_weighted = precision_recall_fscore_support(
                    preserved_targets, preserved_preds, average='weighted', zero_division=0
                )[2]
                
                try:
                    preserved_roc_auc_macro = roc_auc_score(
                        preserved_targets, preserved_preds, average='macro'
                    )
                except ValueError:
                    preserved_roc_auc_macro = float('nan')
                
                preserved_accuracy_macro = np.mean([
                    accuracy_score(preserved_targets[:, i], preserved_preds[:, i])
                    for i in range(preserved_targets.shape[1])
                ])
                
                preserved_metrics = {
                    'subset_accuracy': preserved_subset_accuracy,
                    'precision_macro': preserved_precision_macro,
                    'recall_macro': preserved_recall_macro,
                    'f1_macro': preserved_f1_macro,
                    'precision_weighted': preserved_precision_weighted,
                    'recall_weighted': preserved_recall_weighted,
                    'f1_weighted': preserved_f1_weighted,
                    'accuracy_macro': preserved_accuracy_macro,
                    'roc_auc_macro': preserved_roc_auc_macro
                }
        
        return {
            'loss': loss,
            'subset_accuracy': subset_accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'accuracy_macro': accuracy_macro,
            'roc_auc_macro': roc_auc_macro,
            'per_class_metrics': per_class_metrics,
            'preserved_metrics': preserved_metrics
        }


class DetectionEvaluator:
    """Evaluator for detection models"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """
        Evaluate detection model using COCO metrics
        
        Args:
            model: Detection model to evaluate
            dataloader: Test data loader
            
        Returns:
            Dictionary containing mAP metrics
        """
        model.eval()
        
        metrics = {
            'main': MeanAveragePrecision(
                box_format='xyxy',
                class_metrics=True
            ).to(self.device),
            '50': MeanAveragePrecision(
                box_format='xyxy',
                class_metrics=True,
                iou_thresholds=[0.5]
            ).to(self.device),
            '75': MeanAveragePrecision(
                box_format='xyxy',
                class_metrics=True,
                iou_thresholds=[0.75]
            ).to(self.device)
        }
        
        for images, targets in dataloader:
            images = [img.to(self.device) for img in images]
            preds = model(images)
            
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            for metric in metrics.values():
                metric.update(preds, targets)
        
        results = metrics['main'].compute()
        results['map_50_per_class'] = metrics['50'].compute()['map_per_class']
        results['map_75_per_class'] = metrics['75'].compute()['map_per_class']
        
        return results
    
    def print_report(self, results: Dict[str, Any], class_names: List[str]):
        """Print detailed evaluation report"""
        print("\n" + "="*80)
        print("DETECTION EVALUATION REPORT")
        print("="*80)
        
        def to_float(x):
            return float(x) if x is not None else 0.0
        
        print(f"\nmAP50-95: {to_float(results['map']):.4f}")
        print(f"mAP50:    {to_float(results.get('map_50', results['map_50_per_class'].mean())):.4f}")
        print(f"mAP75:    {to_float(results.get('map_75', results['map_75_per_class'].mean())):.4f}")
        print(f"mAR100:   {to_float(results.get('mar_100', 0)):.4f}")
        
        print("\n--- Per-Class mAP (50-95) ---")
        for i, class_name in enumerate(class_names):
            if i < len(results["map_per_class"]):
                print(f"{class_name:25s}: {float(results['map_per_class'][i]):.4f}")
        
        print("\n--- Per-Class mAP50 ---")
        for i, class_name in enumerate(class_names):
            if i < len(results["map_50_per_class"]):
                print(f"{class_name:25s}: {float(results['map_50_per_class'][i]):.4f}")
        
        print("\n--- Per-Class mAP75 ---")
        for i, class_name in enumerate(class_names):
            if i < len(results["map_75_per_class"]):
                print(f"{class_name:25s}: {float(results['map_75_per_class'][i]):.4f}")
        
        print("\n" + "="*80 + "\n")