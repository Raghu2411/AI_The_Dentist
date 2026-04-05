"""
Grad-CAM Visualization
Generates ONE Grad-CAM per image showing ALL positive predicted classes
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from typing import List, Optional, Tuple


class GradCAM:
    """Gradient weighted Class Activation Mapping"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[:, class_idx]
        target.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        if activations.dim() == 3 and activations.shape[1] == 197:
            activations = activations[:, 1:, :]
            gradients = gradients[:, 1:, :]
            weights = gradients.mean(dim=2, keepdim=True)
            cam = (weights * activations).sum(dim=2)
            side = int(cam.shape[1] ** 0.5)
            cam = cam.reshape(side, side).detach().cpu().numpy()
        elif activations.dim() == 4:
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam).squeeze().cpu().numpy()
            cam = cv2.resize(cam, (224, 224))
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")
        
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class GradCAMVisualizer:
    """Generates and saves combined Grad-CAM visualizations"""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        target_layer: nn.Module,
        device: torch.device,
        output_dir: str,
        preserved_classes: List[str]
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.output_dir = os.path.join(output_dir, "gradcam_results")
        os.makedirs(self.output_dir, exist_ok=True)
        self.preserved_classes = preserved_classes
        
        self.gradcam = GradCAM(model, target_layer)
        self.model.eval()
    
    def generate_for_dataset(
        self,
        dataloader: DataLoader,
        test_json: str,
        test_img_dir: str,
        class_names: List[str],
        preserved_classes: List[str],
        max_images: Optional[int] = None
    ):
        print(f"\nGenerating COMBINED Grad-CAM visualizations for {self.model_name}...")
        print(f"Preserved classes: {preserved_classes}")
        
        coco = COCO(test_json)
        
        processed = 0
        for idx, (image, label, filename) in enumerate(dataloader):
            if max_images and processed >= max_images:
                break
            
            image = image.to(self.device)
            output = self.model(image)
            probs = torch.sigmoid(output).squeeze().detach().cpu().numpy()
            
            positive_classes = np.where(probs > 0.5)[0]
            
            positive_preserved_classes = [
                (idx, class_names[idx], probs[idx])
                for idx in positive_classes
                if class_names[idx] in preserved_classes
            ]
            
            if len(positive_preserved_classes) == 0:
                continue
            
            print(f"\nProcessing: {filename[0]}")
            print(f" Positive classes: {[name for _, name, _ in positive_preserved_classes]}")
            
            self._generate_combined_visualization(
                image=image,
                positive_classes=positive_preserved_classes,
                filename=filename[0],
                test_img_dir=test_img_dir,
                coco=coco
            )
            processed += 1
        
        print(f"\nGenerated {processed} combined Grad-CAM visualizations → {self.output_dir}")
    
    def _generate_combined_visualization(
        self,
        image: torch.Tensor,
        positive_classes: List[Tuple[int, str, float]],
        filename: str,
        test_img_dir: str,
        coco: COCO
    ):
        """Generate ONE combined visualization for all positive classes"""
        
        orig_img = cv2.imread(os.path.join(test_img_dir, filename))
        if orig_img is None:
            return
        orig_img = cv2.resize(orig_img, (224, 224))
        
        combined_heatmap = np.zeros((224, 224), dtype=np.float32)
        
        for class_idx, class_name, prob in positive_classes:
            print(f" Generating heatmap for: {class_name} ({prob:.2f})")
            heatmap = self.gradcam.generate(image, class_idx)
            
            if heatmap.shape != (224, 224):
                print(f"\n Resizing heatmap from {heatmap.shape} to (224, 224)")
                heatmap = cv2.resize(heatmap, (224, 224))
            
            combined_heatmap += heatmap * prob
        
        combined_heatmap = (combined_heatmap - combined_heatmap.min()) / \
                          (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
        
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * combined_heatmap), 
            cv2.COLORMAP_JET
        )
        
        overlay = cv2.addWeighted(orig_img, 0.5, heatmap_colored, 0.5, 0)
        
        bbox_img = self._draw_bounding_boxes(
            orig_img.copy(), 
            filename, 
            coco,
            positive_classes
        )
        
        combined = np.hstack((bbox_img, overlay))
        
        class_labels = [f"{name}({prob:.2f})" for _, name, prob in positive_classes]
        label_text = "Predictions: " + ", ".join(class_labels)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        overlay_bg = combined.copy()
        cv2.rectangle(overlay_bg, (5, 5), (min(text_w + 15, combined.shape[1] - 5), text_h + baseline + 10), (0, 0, 0), -1)
        combined = cv2.addWeighted(combined, 0.7, overlay_bg, 0.3, 0)
        
        cv2.putText(combined, label_text, (10, text_h + 10), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        cv2.putText(combined, "Ground Truth", (10, combined.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        legend_y = combined.shape[0] - 30
        cv2.putText(combined, "Green=TP", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(combined, "Blue=FN", (70, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        
        cv2.putText(combined, "Combined Grad-CAM", (224 + 10, combined.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        
        class_names_str = "_".join([name for _, name, _ in positive_classes])
        save_name = f"{os.path.splitext(filename)[0]}_combined_{class_names_str}_gradcam.jpg"
        save_path = os.path.join(self.output_dir, save_name)
        cv2.imwrite(save_path, combined)
        print(f" Saved: {save_name}")
    
    def _draw_bounding_boxes(
        self,
        img: np.ndarray,
        filename: str,
        coco: COCO,
        positive_classes: List[Tuple[int, str, float]]
    ) -> np.ndarray:
        """Draw bounding boxes with TP/FN color coding for all predictions"""
        
        predicted_class_names = [name for _, name, _ in positive_classes]
        
        img_ids = [
            img_info['id'] for img_info in coco.dataset['images']
            if img_info['file_name'] == filename
        ]
        
        if not img_ids:
            return img
        
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_ids[0]))
        img_info = coco.loadImgs(img_ids[0])[0]
        orig_w, orig_h = img_info['width'], img_info['height']
        
        scale_x = 224 / orig_w
        scale_y = 224 / orig_h
        
        boxes_to_label = []
        
        for ann in anns:
            cat = coco.loadCats(ann['category_id'])[0]['name']
            
            if cat not in self.preserved_classes:
                continue
            
            if cat in predicted_class_names:
                color = (0, 255, 0)  # Green - TP
                status = "TP"
                print(f' TP: "{cat}" was predicted')
            else:
                color = (255, 0, 0)  # Blue - FN
                status = "FN"
                print(f'  FN: "{cat}" was NOT predicted')
            
            x, y, w, h = ann['bbox']
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            boxes_to_label.append((x, y, w, h, cat, color, status))
        
        used_positions = []
        
        for x, y, w, h, cat, color, status in boxes_to_label:
            label = cat
            if len(label) > 10:
                abbrev = {
                    'periapical radiolucency': 'periapical',
                    'rc-treated': 'rc-treat',
                    'restoration': 'restore',
                    'root-stump': 'r-stump'
                }
                label = abbrev.get(label, label[:8] + '..')
            
            label_with_status = f"{label}({status})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            
            (text_w, text_h), baseline = cv2.getTextSize(label_with_status, font, font_scale, thickness)
            
            possible_positions = [
                (x, y - 5, 'above'),
                (x, y + h + text_h + 5, 'below'),
                (x + 2, y + text_h + 2, 'inside_top'),
                (x + 2, y + h - 5, 'inside_bottom')
            ]
            
            label_x, label_y = x, y - 5
            for pos_x, pos_y, pos_type in possible_positions:
                overlap = False
                for used_x, used_y, used_w, used_h in used_positions:
                    if (abs(pos_x - used_x) < text_w + 5 and abs(pos_y - used_y) < text_h + 5):
                        overlap = True
                        break
                if not overlap:
                    label_x, label_y = pos_x, pos_y
                    break
            
            padding = 2
            cv2.rectangle(img, (label_x - padding, label_y - text_h - padding), (label_x + text_w + padding, label_y + baseline), color, -1)
            cv2.putText(img, label_with_status, (label_x, label_y - 1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            used_positions.append((label_x, label_y, text_w, text_h))
        
        return img


def generate_gradcam_for_model(
    model_name: str,
    model_path: str,
    test_json: str,
    test_img_dir: str,
    class_names: List[str],
    preserved_classes: List[str],
    device: torch.device,
    output_dir: str,
    transform,
    max_images: Optional[int] = None
):
    from datasets import DatasetFactory
    from model_factory import ModelFactory
    from torch.utils.data import DataLoader
    
    print(f"\n{'='*80}")
    print(f"GENERATING COMBINED GRAD-CAM VISUALIZATIONS: {model_name.upper()}")
    print(f"{'='*80}")
    
    test_dataset = DatasetFactory.create_classification_dataset(test_json, test_img_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = ModelFactory.create_model(model_name, len(class_names), device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    target_layer = ModelFactory.get_gradcam_layer(model_name, model)
    
    visualizer = GradCAMVisualizer(
        model=model,
        model_name=model_name,
        target_layer=target_layer,
        device=device,
        output_dir=output_dir,
        preserved_classes=preserved_classes
    )
    
    visualizer.generate_for_dataset(
        dataloader=test_loader,
        test_json=test_json,
        test_img_dir=test_img_dir,
        class_names=class_names,
        preserved_classes=preserved_classes,
        max_images=max_images
    )