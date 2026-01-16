# train_other_models.py

# #!/bin/bash
# set -e
# echo " Installation for Torchvision, YOLO & Florence-2 Models "
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install ultralytics torchmetrics Pillow opencv-python scikit-learn
# pip install -U transformers peft accelerate sentencepiece
# echo "Installing bitsandbytes from source..."
# pip uninstall -y bitsandbytes
# pip install --no-binary :all: bitsandbytes
# echo "Setup complete."

import torch, numpy as np, os, json, warnings, gc
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
                                          retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# import flash_attn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from ultralytics import YOLO
from transformers import (AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments,
                          EarlyStoppingCallback)
import glob
from pathlib import Path
import shutil

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRESERVED_CLASSES = ['calculus', 'caries', 'crown', 'impacted', 'implant',
      'periapical radiolucency', 'rc-treated', 'restoration', 'root-stump']

def collate_fn(batch): return tuple(zip(*batch))

class DetectionOPGDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f: self.data = json.load(f)
        self.img_dir, self.transform = img_dir, transform
        self.categories = sorted(self.data['categories'], key=lambda x: x['id'])
        self.cat_id_to_label = {c['id']: i for i, c in enumerate(self.categories)}
        self.class_names = [c['name'] for c in self.categories]
        self.annotations_by_image = {img['id']: [] for img in self.data['images']}
        for ann in self.data['annotations']: self.annotations_by_image[ann['image_id']].append(ann)
        self.image_ids_with_ann = [img_id for img_id, anns in self.annotations_by_image.items() if anns]
        self.images_map = {img['id']: img for img in self.data['images']}
        print(f"Loaded {os.path.basename(json_file)}: {len(self.image_ids_with_ann)} images, {len(self.class_names)} classes.")

    def __len__(self): return len(self.image_ids_with_ann)
    def __getitem__(self, idx):
        img_id = self.image_ids_with_ann[idx]
        img_info = self.images_map[img_id]
        image = Image.open(os.path.join(self.img_dir, img_info['file_name'])).convert('RGB')
        boxes, labels = [], []
        for ann in self.annotations_by_image[img_id]:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(self.cat_id_to_label[ann['category_id']])
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32), "labels": torch.as_tensor(labels, dtype=torch.int64)}
        if self.transform: image = self.transform(image)
        return image, target

def print_evaluation_report(results, all_class_names):
    print("\n Classification Report (Summary)")
    print(f"  mAP (IoU=0.50:0.95): {results.get('map', -1):.4f}")
    print(f"  mAP50 (IoU=0.50):    {results.get('map_50', -1):.4f}")
    print(f"  mAP75 (IoU=0.75):    {results.get('map_75', -1):.4f}")
    print(f"  mAR (maxDets=100):   {results.get('mar_100', -1):.4f}")

    print("\n Per-Class Metrics")
    print(f"{'Class Name':<30} | {'mAP':<8} | {'mAP50':<8} | {'mAP75':<8}")
    print("-" * 65)

    def get_val(source, idx):
        if source is None: return -1.0
        val = source[idx] if idx < len(source) else -1
        if hasattr(val, 'item'): val = val.item()
        return val

    per_class_map = results.get('map_per_class', [])
    per_class_50 = results.get('map_50_per_class', [])
    per_class_75 = results.get('map_75_per_class', [])

    for i, name in enumerate(all_class_names):
        ap_main = get_val(per_class_map, i)
        ap_50 = get_val(per_class_50, i)
        ap_75 = get_val(per_class_75, i)

        s_main = f"{ap_main:.4f}" if ap_main != -1 else "N/A"
        s_50 = f"{ap_50:.4f}" if ap_50 != -1 else "N/A"
        s_75 = f"{ap_75:.4f}" if ap_75 != -1 else "N/A"

        print(f"{name:<30} | {s_main:<8} | {s_50:<8} | {s_75:<8}")

    print("-" * 65)

    indices = [i for i, n in enumerate(all_class_names) if n in PRESERVED_CLASSES]
    
    def calc_preserved_mean(key):
        data = results.get(key, [])
        if hasattr(data, 'tolist'): data = data.tolist()
        if not data or len(data) <= 1: return "Not Available"
        
        valid_vals = [data[i] for i in indices if i < len(data) and data[i] != -1]
        return f"{sum(valid_vals)/len(valid_vals):.4f}" if valid_vals else "0.0000"

    print("\n Classification Report (Preserved Classes Average)")
    print(f"  mAP (IoU=0.50:0.95): {calc_preserved_mean('map_per_class')}")
    print(f"  mAP50 (IoU=0.50):    {calc_preserved_mean('map_50_per_class')}")
    print(f"  mAP75 (IoU=0.75):    {calc_preserved_mean('map_75_per_class')}")
    print("-" * 45)

@torch.no_grad()
def evaluate_torchvision(model, dataloader, device):
    model.eval()
    metrics = {k: MeanAveragePrecision(box_format='xyxy', class_metrics=True, **v).to(device) for k, v in {'main': {}, '50': {'iou_thresholds': [0.5]}, '75': {'iou_thresholds': [0.75]}}.items()}
    for images, targets in dataloader:
        preds = model([img.to(device) for img in images])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for m in metrics.values(): m.update(preds, targets)
    res = metrics['main'].compute()
    res['map_50_per_class'] = metrics['50'].compute()['map_per_class']
    res['map_75_per_class'] = metrics['75'].compute()['map_per_class']
    return res

def train_torchvision_model(model_name, loaders, num_classes, class_names, device):
    if model_name == 'faster_rcnn':
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes + 1)
    else:
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        model.head.classification_head = RetinaNetClassificationHead(model.head.classification_head.conv[0][0].in_channels, model.head.classification_head.num_anchors, num_classes)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, verbose=True)
    best_map, no_improve, save_path = 0.0, 0, f'best_{model_name}.pth'
    
    for epoch in range(100):
        model.train()
        for images, targets in loaders['train']:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            sum(loss for loss in model(images, targets).values()).backward()
            optimizer.step()
        val_map = evaluate_torchvision(model, loaders['valid'], device)['map'].item()
        
        print(f"Epoch {epoch+1}/100 | Val mAP: {val_map:.4f}")
        scheduler.step(val_map)
        if val_map > best_map:
            best_map, no_improve = val_map, 0; torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= 10: 
                print("Early stopping.")
                break
    print("\n Evaluating on Test Set")
    model.load_state_dict(torch.load(save_path))
    py_res = {k: v.cpu().tolist() for k, v in evaluate_torchvision(model, loaders['test'], device).items()}
    print_evaluation_report(py_res, class_names)
    os.rename(save_path, f'final_{model_name}.pth')

def train_yolo_model(model_pt, data_yaml, class_names, device):
    model = YOLO(model_pt)
    
    model_variant = Path(model_pt).stem 
    run_name = f'{model_variant}_dental_run'

    model.train(data=data_yaml, epochs=100, patience=10, device=device, name=run_name)

    runs_dir = Path('runs/detect')
    if not runs_dir.exists():
        print("Error: Training output directory not found!")
        return
    
    run_dirs = sorted(runs_dir.glob(f'{run_name}*'), key=os.path.getmtime)
    if not run_dirs:
        print(f"Error: No training run directories found for {run_name}!")
        return
    
    latest_run = run_dirs[-1]
    best_weight_path = latest_run / 'weights' / 'best.pt'
    
    if not best_weight_path.exists():
        print(f"Error: Best model not found at {best_weight_path}")
        return
    
    print(f"Loading best model from: {best_weight_path}")
    best_model = YOLO(best_weight_path)
    
    print(f"\n Evaluating {model_variant.upper()} on Test Set")
    test_res = best_model.val(split='test', plots=False)
    
    report = {
        'map': test_res.box.map, 
        'map_50': test_res.box.map50, 
        'map_75': test_res.box.map75,
        'mar_100': test_res.box.mr, 
        'map_per_class': test_res.box.maps,
        'map_50_per_class': test_res.box.all_ap[:, 0], 
        'map_75_per_class': test_res.box.all_ap[:, 5]
    }
    
    print_evaluation_report(report, class_names)
    
    final_model_path = f'final_{model_variant}.pt'
    shutil.copy(best_weight_path, final_model_path)
    print(f"Final model saved to: {final_model_path}")

class Florence2Dataset(Dataset):
    def __init__(self, ds, proc): self.ds, self.proc = ds, proc
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, tgt = self.ds[idx]
        txt = "<OD>" + "".join([f"{self.ds.class_names[l]}<loc_{int(b[0])}><loc_{int(b[1])}><loc_{int(b[2])}><loc_{int(b[3])}>" for l, b in zip(tgt['labels'], tgt['boxes'])])
        return {k: v.squeeze(0) for k, v in self.proc(text=txt, images=img, return_tensors="pt").items()}

def train_florence2_model(train_ds, val_ds):
    model_id = 'microsoft/Florence-2-large'
    
    # Patch to bypass flash_attn requirement
    import sys
    from unittest.mock import MagicMock
    sys.modules['flash_attn'] = MagicMock()
    sys.modules['flash_attn.bert_padding'] = MagicMock()
    sys.modules['flash_attn.flash_attn_interface'] = MagicMock()
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    # Load with eager attention (no flash_attn)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        quantization_config=bnb_config, 
        attn_implementation="eager"
    )
    
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(r=8, lora_alpha=8, target_modules=["q_proj", "o_proj", "k_proj", "v_proj"])
    model = get_peft_model(model, lora_config)
    
    args = TrainingArguments(
        output_dir="florence2_dental_finetune", num_train_epochs=100,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=4e-5, logging_steps=10, save_strategy="epoch",
        evaluation_strategy="epoch", load_best_model_at_end=True, report_to="none"
    )
    
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    trainer = Trainer(
        model=model, train_dataset=Florence2Dataset(train_ds, proc),
        eval_dataset=Florence2Dataset(val_ds, proc),
        args=args, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    trainer.train()
    print("Florence-2 training complete. Best model saved.")

def main():
    base_dir = '/home/tq24401/llm/filtered_opg_dataset_for_LLM'
    paths = {s: (os.path.join(base_dir, f'{s}/_annotations.coco.json'), os.path.join(base_dir, s)) for s in ['train', 'valid', 'test']}
    yolo_yaml = '/home/tq24401/llm/YOLO/data.yaml'
    
    models_to_train = ['yolov11m', 'yolov9', 'faster_rcnn', 'retinanet']
    
    tr, vtr = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(p=0.5)]), T.Compose([T.ToTensor()])
    
    for model_name in models_to_train:
        print(f"\n{'='*60}\nSTARTING TRAINING FOR {model_name.upper()}\n{'='*60}")
        
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        dsets = {s: DetectionOPGDataset(paths[s][0], paths[s][1], transform=(tr if s=='train' else vtr)) 
                 for s in ['train', 'valid', 'test']}
        
        if model_name in ['faster_rcnn', 'retinanet']:
            loaders = {s: DataLoader(d, batch_size=2, collate_fn=collate_fn, num_workers=2, shuffle=(s=='train')) 
                      for s, d in dsets.items()}
            train_torchvision_model(model_name, loaders, len(dsets['train'].class_names), dsets['train'].class_names, device)

            del loaders
        elif model_name == 'yolov9': 
            train_yolo_model('yolov9c.pt', yolo_yaml, dsets['train'].class_names, device)
        elif model_name == 'yolov11m':
            train_yolo_model('yolo11m.pt', yolo_yaml, dsets['train'].class_names, device)
        elif model_name == 'florence2': 
            train_florence2_model(dsets['train'], dsets['valid'])
        
        del dsets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

if __name__ == "__main__":
    main()