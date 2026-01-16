# #!/bin/bash
# set -e
# echo " Detectron2 Installation for ViTDet "
# pip uninstall -y detectron2
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install opencv-python Pillow
# if [ ! -d "detectron2" ]; then
#     git clone https://github.com/facebookresearch/detectron2.git
# fi
# cd detectron2 && pip install -e . && cd ..
# echo "Detectron2 setup complete. Need to move train_detectron2.py into detectron2/ and run it from there."

import os
import torch
import warnings
import logging
import json
import numpy as np
from detectron2.engine import DefaultTrainer, DefaultPredictor, hooks
from detectron2.utils.events import get_event_storage
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

warnings.filterwarnings("ignore", category=UserWarning)

PRESERVED_CLASSES = ['calculus', 'caries', 'crown', 'impacted', 'implant',
      'periapical radiolucency', 'rc-treated', 'restoration', 'root-stump']

# class EarlyStoppingHook(hooks.HookBase):
#     """
#     A hook to implement early stopping.
#     It stops training if the specified metric does not improve for a
#     given number of evaluations.
#     """
#     def __init__(self, patience, monitor="bbox/AP", rule="max"):
#         """
#         Args:
#             patience (int): Number of evaluations to wait for improvement.
#             monitor (str): The validation metric to monitor. Must be in storage.
#             rule (str): "max" or "min". If "max", training stops if the metric
#                         stops increasing. If "min", training stops if the
#                         metric stops decreasing.
#         """
#         self.patience = patience
#         self.monitor = monitor
#         self.rule = rule
#         self._logger = logging.getLogger(__name__)

#         # Initialize state
#         self.best_metric = -float('inf') if rule == "max" else float('inf')
#         self.patience_counter = 0

#     def after_step(self):
#         # This hook runs after every step. We only act if an evaluation
#         # has just been performed (i.e., the metric is in the storage).
#         storage = get_event_storage()
        
#         # Check if the monitored metric is in the latest iteration's results
#         if self.monitor in storage.latest():
#             latest_metric = storage.latest()[self.monitor]
#             if isinstance(latest_metric, tuple):
#                 latest_metric = latest_metric[0]

#             metric_improved = False
#             if self.rule == "max":
#                 if latest_metric > self.best_metric:
#                     self.best_metric = latest_metric
#                     metric_improved = True
#             else: # rule == "min"
#                 if latest_metric < self.best_metric:
#                     self.best_metric = latest_metric
#                     metric_improved = True

#             if metric_improved:
#                 # Reset counter if metric improved
#                 self.patience_counter = 0
#             else:
#                 # Increment counter if metric did not improve
#                 self.patience_counter += 1
#                 self._logger.info(
#                     f"[EarlyStoppingHook] Metric '{self.monitor}' did not improve. "
#                     f"Patience counter: {self.patience_counter} / {self.patience}"
#                 )

#             # Check if training should stop
#             if self.patience_counter >= self.patience:
#                 self._logger.info(
#                     f"[EarlyStoppingHook] Metric '{self.monitor}' did not improve for {self.patience} evaluations."
#                 )
#                 self._logger.info(f"Best metric was: {self.best_metric:.4f}. Stopping training.")
#                 # We stop training by raising an exception that the SimpleTrainer will catch.
#                 raise StopIteration("Early stopping triggered.")

#     def after_train(self):
#         # Reset state after training is finished
#         self.patience_counter = 0
#         self.best_metric = -float('inf') if self.rule == "max" else float('inf')

# Custom Trainer to add periodic evaluation and best model saving
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        
        # This hook saves the best model
        hooks_list.append(hooks.BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD, self.checkpointer, "bbox/AP", "max"
        ))
        # hooks_list.append(EarlyStoppingHook(
        #     patience=7, 
        #     monitor="bbox/AP", 
        #     rule="max"
        # ))
        
        return hooks_list

def print_preserved_class_report(eval_json_path, all_class_names):
    print("\n--- Classification Report (Preserved Classes) ---")
    if not os.path.exists(eval_json_path):
        print(f"  Evaluation JSON not found at {eval_json_path}")
        return

    with open(eval_json_path, 'r') as f:
        eval_data = json.load(f)


def setup_vitdet_config(cfg, num_classes, weights_path):
    """
    Manually configure ViTDet settings based on the config file parameters.
    Replicates the mask_rcnn_vitdet_b_100ep.py configuration.
    """

    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_vit_fpn_backbone"
    
    # ViT-B settings (Base model)
    cfg.MODEL.VIT = get_cfg()
    cfg.MODEL.VIT.EMBED_DIM = 768
    cfg.MODEL.VIT.DEPTH = 12
    cfg.MODEL.VIT.NUM_HEADS = 12
    cfg.MODEL.VIT.MLP_RATIO = 4.0
    cfg.MODEL.VIT.QKV_BIAS = True
    cfg.MODEL.VIT.DROP_PATH_RATE = 0.1
    cfg.MODEL.VIT.WINDOW_SIZE = 14
    cfg.MODEL.VIT.OUT_FEATURES = ["s1", "s2", "s3", "s4"]
    
    # FPN settings
    cfg.MODEL.FPN.IN_FEATURES = ["s1", "s2", "s3", "s4"]
    cfg.MODEL.FPN.OUT_CHANNELS = 256
    
    # RPN settings
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    # ROI Head settings
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    
    # Mask head settings
    cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    
    # Weights
    if os.path.exists(weights_path):
        cfg.MODEL.WEIGHTS = weights_path
        print(f"Using pretrained ViTDet weights: {weights_path}")
    else:
        # MAE pretrained ViT backbone
        cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth"
        print(f"ViTDet weights not found at {weights_path}")
        print("Using MAE pretrained ViT backbone")
    
    return cfg


def main():
    base_dir = '/home/tq24401/llm/filtered_opg_dataset_for_LLM'
    dataset_prefix = 'dental_opg'
    
    with open(os.path.join(base_dir, 'train/_annotations.coco.json')) as f:
        class_names = [cat['name'] for cat in json.load(f)['categories']]
    
    # Register datasets
    paths = {s: (os.path.join(base_dir, f'{s}/_annotations.coco.json'), os.path.join(base_dir, s)) 
             for s in ['train', 'valid', 'test']}
    for d, (json_p, img_d) in [("train", paths['train']), ("val", paths['valid']), ("test", paths['test'])]:
        name = f"{dataset_prefix}_{d}"
        if name in DatasetCatalog.list(): 
            DatasetCatalog.remove(name)
        register_coco_instances(name, {}, json_p, img_d)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Set weights
    weights_path = "/home/tq24401/Dental/detectron2/projects/ViTDet/model_final_61ccd1.pkl"
    if os.path.exists(weights_path):
        print(f"\n??  WARNING: Using standard Mask R-CNN config, but ViTDet weights provided.")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        print(f"Using standard Mask R-CNN COCO weights.")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        print(f"Using standard Mask R-CNN COCO pretrained weights")
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = (f"{dataset_prefix}_train",)
    cfg.DATASETS.TEST = (f"{dataset_prefix}_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Model configuration
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Solver configuration
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001  # Lower learning rate for fine-tuning
    cfg.SOLVER.MAX_ITER = 20000
    
    # Learning rate schedule - Multi-step decay
    cfg.SOLVER.STEPS = (15000, 18000)  # Reduce LR at 75% and 90% of training
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    
    # Output directory
    cfg.OUTPUT_DIR = "./output_detectron2_maskrcnn"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Evaluation period
    cfg.TEST.EVAL_PERIOD = 500
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print("STARTING TRAINING FOR DETECTRON2 MASK R-CNN")
    print(f"{'='*60}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(DatasetCatalog.get(f'{dataset_prefix}_train'))}")
    print(f"Validation samples: {len(DatasetCatalog.get(f'{dataset_prefix}_val'))}")
    print(f"Test samples: {len(DatasetCatalog.get(f'{dataset_prefix}_test'))}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Base learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # Train
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluate on test set
    print("\n Evaluating Best Model on Test Set ")
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    if os.path.exists(best_model_path):
        cfg.MODEL.WEIGHTS = best_model_path
        print(f"Loading best model from: {best_model_path}")
    else:
        print(f"Best model not found at {best_model_path}, using final model")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(f"{dataset_prefix}_test", 
                              output_dir=os.path.join(cfg.OUTPUT_DIR, "test_eval"))
    test_loader = build_detection_test_loader(cfg, f"{dataset_prefix}_test")
    test_results_raw = inference_on_dataset(predictor.model, test_loader, evaluator)
    
    print("\n Classification Report (All Classes)")
    print(f"  mAP (IoU=0.50:0.95) : {test_results_raw['bbox']['AP']:.4f}")
    print(f"  mAP50 (IoU=0.50)    : {test_results_raw['bbox']['AP50']:.4f}")
    print(f"  mAP75 (IoU=0.75)    : {test_results_raw['bbox']['AP75']:.4f}")
    print(f"  APs (Small Objects) : {test_results_raw['bbox']['APs']:.4f}")
    print(f"  APm (Medium Objects): {test_results_raw['bbox']['APm']:.4f}")
    print(f"  APl (Large Objects) : {test_results_raw['bbox']['APl']:.4f}")
    
    eval_json_path = os.path.join(cfg.OUTPUT_DIR, "test_eval", "coco_instances_results.json")
    print_preserved_class_report(eval_json_path, class_names)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()