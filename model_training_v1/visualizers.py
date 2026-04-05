"""
Visualization Utilities
Handles plotting and visualization tasks
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import List, Dict, Any


class LearningCurveVisualizer:
    """Visualizes learning curves for model training"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_classification_curves(
        self,
        history: List[Dict[str, Any]],
        model_name: str
    ):
        """Plot loss and F1 curves for classification"""
        df = pd.DataFrame(history)
        
        plt.figure(figsize=(7, 5))
        plt.plot(df["Epoch"], df["Train_Loss"], 
                label="Train Loss", marker='o', color='blue')
        plt.plot(df["Epoch"], df["Val_Loss"], 
                label="Validation Loss", marker='s', color='orange')
        plt.title(f"Loss Curve: {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self.output_dir, f"loss_curve_{model_name}.png"),
            dpi=200,
            bbox_inches='tight'
        )
        plt.close()
        
        plt.figure(figsize=(7, 5))
        plt.plot(df["Epoch"], df["Train_F1_Weighted"], 
                label="Train F1-Weighted", marker='o', color='blue')
        plt.plot(df["Epoch"], df["Val_F1_Weighted"], 
                label="Validation F1-Weighted", marker='s', color='green')
        plt.title(f"F1-Weighted Learning Curve: {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("F1-Weighted Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self.output_dir, f"f1_curve_{model_name}.png"),
            dpi=200,
            bbox_inches='tight'
        )
        plt.close()
        
        print(f"Saved learning curves to: {self.output_dir}")
    
    def plot_detection_curves(
        self,
        train_losses: List[float],
        val_maps: List[float],
        test_maps: List[float],
        model_name: str
    ):
        """Plot training curves for detection models"""
        curves_dir = os.path.join(self.output_dir, "training_curves")
        os.makedirs(curves_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name} Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, "training_loss.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(val_maps, label="Validation mAP50-95", 
                color="orange", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.title(f"{model_name} Validation mAP Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, "validation_map.png"))
        plt.close()
        
        print(f"Learning curves saved → {curves_dir}")


class ConfusionMatrixVisualizer:
    """Visualizes confusion matrices"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        save_name: str = "confusion_matrix.png"
    ):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues"
        )
        plt.title("Detection Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("Ground-Truth Class")
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        
        print(f"📈 Confusion matrix saved → {save_path}")


class MetricsTableGenerator:
    """Generates metrics tables"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_detection_metrics_table(
        self,
        detection_results: List[tuple],
        class_names: List[str],
        save_name: str = "metrics.csv"
    ):
        """Generate per-class detection metrics table"""
        per_class = {
            cls_id: {"TP": 0, "FP": 0, "FN": 0}
            for cls_id in range(len(class_names))
        }
        
        for gt, pred in detection_results:
            if gt is not None and pred is not None:
                if gt == pred:
                    per_class[gt]["TP"] += 1
                else:
                    per_class[pred]["FP"] += 1
                    per_class[gt]["FN"] += 1
            elif gt is None and pred is not None:
                per_class[pred]["FP"] += 1
            elif gt is not None and pred is None:
                per_class[gt]["FN"] += 1
        
        rows = []
        for cls_id, stats in per_class.items():
            TP = stats["TP"]
            FP = stats["FP"]
            FN = stats["FN"]
            
            precision = TP / (TP + FP + 1e-9)
            recall = TP / (TP + FN + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            
            rows.append([
                class_names[cls_id], TP, FP, FN,
                round(precision, 4),
                round(recall, 4),
                round(f1, 4)
            ])
        
        df = pd.DataFrame(rows, columns=[
            "Class", "TP", "FP", "FN", "Precision", "Recall", "F1"
        ])
        
        save_path = os.path.join(self.output_dir, save_name)
        df.to_csv(save_path, index=False)
        print(f"📊 Metrics table saved → {save_path}")
        
        return df
