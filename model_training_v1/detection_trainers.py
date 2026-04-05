"""
Detection models training
Implements RetinaNet and YOLO training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import os
from pathlib import Path
import shutil

from training_strategies import DetectionTrainingStrategy
from evaluators import DetectionEvaluator
from visualizers import LearningCurveVisualizer


class RetinaNetTrainingStrategy(DetectionTrainingStrategy):
    """Training strategy for RetinaNet"""
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        **kwargs
    ) -> nn.Module:
        """Train RetinaNet model"""
        print(f"\nTraining RetinaNet\n")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.1
        )
        
        best_map = 0
        patience_counter = 0
        train_losses = []
        val_maps = []
        test_maps = []
        
        evaluator = DetectionEvaluator(self.device)
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer)
            scheduler.step()
            
            print(f"Epoch [{epoch}/{self.config.training.num_epochs}]  "
                  f"Loss: {train_loss:.4f}")
            train_losses.append(train_loss)
            
            val_results = evaluator.evaluate(model, val_loader)
            current_map = float(val_results["map"])
            print(f"Validation mAP50-95: {current_map:.4f}")
            val_maps.append(current_map)
            
            test_results = evaluator.evaluate(model, test_loader)
            test_map = float(test_results["map"])
            print(f"Testing mAP50-95: {test_map:.4f}")
            test_maps.append(test_map)
            
            if current_map > best_map:
                best_map = current_map
                best_path = self.save_model(model, "final_retinanet.pth")
                patience_counter = 0
                print(f"🔥 Best RetinaNet saved (mAP: {best_map:.4f})")
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Patience: {patience_counter}/{self.config.training.patience}")
                
                if patience_counter >= self.config.training.patience:
                    print("\n⛔ Early stopping triggered!")
                    break
            
            self.cleanup_memory()
        
        visualizer = LearningCurveVisualizer(self.output_dir)
        visualizer.plot_detection_curves(
            train_losses, val_maps, test_maps, "RetinaNet"
        )
        
        print(f"\n====== RETINANET TRAINING COMPLETE ======")
        print(f"Best mAP50-95 = {best_map:.4f}\n")
        
        return model
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        
        for images, targets in train_loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: list,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate RetinaNet model"""
        print("\nComputing RetinaNet Test Metrics...\n")
        
        evaluator = DetectionEvaluator(self.device)
        test_results = evaluator.evaluate(model, test_loader)
        
        report = {
            "map": test_results["map"],
            "map_50": test_results["map_50_per_class"].mean(),
            "map_75": test_results["map_75_per_class"].mean(),
            "mar_100": test_results.get("mar_100", None),
            "map_per_class": test_results["map_per_class"],
            "map_50_per_class": test_results["map_50_per_class"],
            "map_75_per_class": test_results["map_75_per_class"],
        }
        
        evaluator.print_report(report, class_names)
        
        return report


class YOLOTrainingStrategy(DetectionTrainingStrategy):
    """Training strategy specifically for YOLO models"""
    
    def train(
        self,
        data_yaml: str,
        model,
        **kwargs
    ):
        """Train YOLO model"""
        print("\n" + "="*80)
        print(f"TRAINING {self.model_name.upper()}")
        print("="*80)
        
        try:
            results = model.train(
                data=data_yaml,
                epochs=self.config.training.num_epochs,
                patience=self.config.training.patience,
                device=self.device,
                name=f'{self.model_name}_training',
                imgsz=self.config.models.yolo_img_size,
                batch=self.config.models.yolo_batch_size,
                workers=1,
                cache=False,
                project=self.output_dir,
                save=True,
                plots=True,
            )
            
            runs_dir = Path(self.output_dir)
            run_dirs = sorted(
                runs_dir.glob(f'{self.model_name}_training*'),
                key=os.path.getmtime
            )
            
            if run_dirs:
                latest_run = run_dirs[-1]
                best_weight_path = latest_run / 'weights' / 'best.pt'
                
                print(f"\nTraining outputs saved to: {latest_run}")
                
                if best_weight_path.exists():
                    final_path = os.path.join(
                        self.output_dir, f'final_{self.model_name}.pt'
                    )
                    shutil.copy(best_weight_path, final_path)
                    print(f"Final model: {final_path}")
                    
                    return str(best_weight_path)
        
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate(
        self,
        model_path: str,
        data_yaml: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate YOLO model on test set"""
        from ultralytics import YOLO
        
        print("\nEvaluating YOLO on TEST set...")
        
        model = YOLO(model_path)
        
        test_results = model.val(
            split='test',
            plots=True,
            save_json=True,
            project=self.output_dir,
            name=f'{self.model_name}_test_evaluation'
        )
        
        test_eval_dir = Path(self.output_dir) / f'{self.model_name}_test_evaluation'
        if test_eval_dir.exists():
            val_folder = test_eval_dir / 'val'
            test_folder = test_eval_dir / 'test'
            if val_folder.exists() and not test_folder.exists():
                val_folder.rename(test_folder)
                print("Renamed output 'val' to 'test'")
        
        report = {
            'map': test_results.box.map,
            'map_50': test_results.box.map50,
            'map_75': test_results.box.map75,
            'mar_100': test_results.box.mr,
            'map_per_class': test_results.box.maps,
            'map_50_per_class': test_results.box.all_ap[:, 0],
            'map_75_per_class': test_results.box.all_ap[:, 5]
        }
        
        print("\nTEST SET EVALUATION RESULTS:")
        self._print_yolo_report(report)
        
        return report
    
    def _print_yolo_report(self, results: Dict[str, Any]):
        """Print YOLO evaluation report"""
        print("\n--- Test Metrics ---")
        print(f"  mAP (IoU=0.50:0.95): {results.get('map', -1):.4f}")
        print(f"  mAP50 (IoU=0.50):    {results.get('map_50', -1):.4f}")
        print(f"  mAP75 (IoU=0.75):    {results.get('map_75', -1):.4f}")
        print(f"  mAR (maxDets=100):   {results.get('mar_100', -1):.4f}")
        print("-" * 45)
