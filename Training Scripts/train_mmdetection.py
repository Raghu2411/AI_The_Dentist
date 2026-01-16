import os
import torch
import warnings
import json
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils.setup_env import setup_cache_size_limit_of_dynamo

warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

PRESERVED_CLASSES = ['calculus', 'caries', 'crown', 'impacted', 'implant',
      'periapical radiolucency', 'rc-treated', 'restoration', 'root-stump']

def print_preserved_class_report(all_results, class_names):
    """
    Parses the MMEngine evaluation dictionary for per-class metrics.
    """
    print("\n Classification Report (Preserved Classes) ")
    map_list, map50_list, map75_list = [], [], []

    for class_name in PRESERVED_CLASSES:
        # MMEngine stores per-class metrics with class names in the key
        map_key = f'coco/AP@.5:.95_class_{class_name}'
        map50_key = f'coco/AP@.5_class_{class_name}'
        map75_key = f'coco/AP@.75_class_{class_name}'

        if map_key in all_results:
            map_list.append(all_results[map_key])
        if map50_key in all_results:
            map50_list.append(all_results[map50_key])
        if map75_key in all_results:
            map75_list.append(all_results[map75_key])

    if not map_list:
        print("  Per-class metrics not found in evaluation results.")
        return

    print(f"  mAP (IoU=0.50:0.95): {np.mean(map_list):.4f}")
    print(f"  mAP50 (IoU=0.50):    {np.mean(map50_list):.4f}")
    print(f"  mAP75 (IoU=0.75):    {np.mean(map75_list):.4f}")
    print("-" * 45)

def main():
    setup_cache_size_limit_of_dynamo()
    
    base_dir = '/home/tq24401/llm/filtered_opg_dataset_for_LLM'
    with open(os.path.join(base_dir, 'train/_annotations.coco.json')) as f:
        coco_data = json.load(f)
    class_names = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]

    #config_file = '/home/tq24401/Dental/mmdetection/configs/mask_dino/mask-dino_swin-l-in22k-pre_8xb2-50e_coco.py'
    config_file = '/home/tq24401/Dental/Detections/mmdetection/configs/dino/dino-4scale_r50_8xb2-24e_coco.py'
    cfg = Config.fromfile(config_file)

    cfg.dataset_type = 'CocoDataset'
    cfg.data_root = base_dir
    for split in ['train', 'val', 'test']:
        split_name = 'valid' if split == 'val' else split
        dataloader_cfg = cfg[f'{split}_dataloader']
        data_cfg = dataloader_cfg.dataset
        cfg.train_dataloader.batch_size = 1
        cfg.val_dataloader.batch_size = 1
        cfg.test_dataloader.batch_size = 1
        cfg.train_dataloader.num_workers = 2
        cfg.val_dataloader.num_workers = 2
        cfg.test_dataloader.num_workers = 2
        cfg.train_dataloader.persistent_workers = False

        data_cfg.ann_file = os.path.join(base_dir, f'{split_name}/_annotations.coco.json')
        data_cfg.data_prefix = dict(img=os.path.join(base_dir, split_name))
        data_cfg.metainfo = dict(classes=tuple(class_names))
        data_cfg.type = 'CocoDataset' 

    # Model Config
    cfg.model.bbox_head.num_classes = len(class_names)
    cfg.load_from = '/home/tq24401/Dental/Detections/mmdetection/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
    cfg.work_dir = './work_dirs/mask_dino_dental'
    cfg.train_cfg.max_epochs = 80  # for early stopping
    cfg.train_cfg.val_interval = 1
    cfg.optim_wrapper.type = 'AmpOptimWrapper'
    cfg.optim_wrapper.loss_scale = 'dynamic'
    
    cfg.optim_wrapper.optimizer.lr = 1e-5
    cfg.param_scheduler = [
        dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
        dict(type='MultiStepLR', by_epoch=True, milestones=[20, 35], gamma=0.1)
    ]

    cfg.default_scope = 'mmdet'
    
    cfg.val_evaluator = dict(
    type='CocoMetric',
    ann_file=cfg.val_dataloader.dataset.ann_file,
    metric='bbox',
    classwise=True
    )
    
    cfg.default_hooks.logger.interval = 50
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP', # Save the best model based on mAP
        rule='greater'
    )
    # Add Early Stopping Hook
    cfg.default_hooks.early_stopping = dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=5,
        rule='greater'
    )
    
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(cfg.model, 'backbone'):
        cfg.model.backbone.with_cp = True
    
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
    
    print(f"\n{'='*60}\nSTARTING TRAINING FOR MMDET_MASKDINO\n{'='*60}")
    torch.cuda.empty_cache()
    runner = Runner.from_cfg(cfg)
    runner.train()

    # Final Evaluation on the Test Set
    print("\n Evaluating Best Model on Test Set")
    best_model_path = os.path.join(cfg.work_dir, 'best_coco_bbox_mAP.pth')
    if not os.path.exists(best_model_path):
        print("Could not find the best model, using last checkpoint.")
        checkpoints = [f for f in os.listdir(cfg.work_dir) if f.startswith('epoch') and f.endswith('.pth')]
        if not checkpoints:
             print("No checkpoints found. Skipping test evaluation.")
             return
        best_model_path = os.path.join(cfg.work_dir, sorted(checkpoints)[-1])
    
    cfg.load_from = best_model_path
    
    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=cfg.test_dataloader.dataset.ann_file,
        metric='bbox',
        classwise=True 
    )
    
    runner = Runner.from_cfg(cfg)
    eval_results = runner.test()
    
    print("\n Classification Report (All Classes) ")
    print(f"  mAP (IoU=0.50:0.95): {eval_results.get('coco/bbox_mAP', -1):.4f}")
    print(f"  mAP50 (IoU=0.50):    {eval_results.get('coco/bbox_mAP_50', -1):.4f}")
    print(f"  mAP75 (IoU=0.75):    {eval_results.get('coco/bbox_mAP_75', -1):.4f}")

    print_preserved_class_report(eval_results, class_names)
    print(f"\nBest model and logs are in: {cfg.work_dir}")

if __name__ == "__main__":
    main()