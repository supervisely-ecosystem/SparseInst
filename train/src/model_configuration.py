from detectron2.config import get_cfg
from sparseinst import add_sparse_inst_config
from detectron2.engine import DefaultTrainer
import os
from detectron2.data import MetadataCatalog, build_detection_train_loader
from typing import Any, Dict, List, Set
import torch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)
from sparseinst import COCOMaskEvaluator
import detectron2.utils.comm as comm
from detectron2.solver.build import maybe_add_gradient_clipping
import itertools
import supervisely as sly
from detectron2.engine.hooks import HookBase
from supervisely.nn.training import train_logger




class IterationHook(HookBase):
    def before_step(self):
        current_iter = self.trainer.iter
        if current_iter == 0:
            self.trainer.sly_logger.train_started(total_epochs=self.trainer.cfg.SOLVER.MAX_ITER)

    def after_step(self):
        current_iter = self.trainer.iter
        self.trainer.sly_logger.epoch_finished()
        if current_iter == self.trainer.cfg.SOLVER.MAX_ITER - 1:
            self.trainer.sly_logger.train_finished()


class Trainer(DefaultTrainer):
    def __init__(self, cfg, sly_logger):
        super().__init__(cfg)
        self.sly_logger = sly_logger

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOMaskEvaluator(dataset_name, ("segm", ), True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            # for transformer
            if "patch_embed" in key or "cls_token" in key:
                weight_decay = 0.0
            if "norm" in key:
                weight_decay = 0.0
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full  model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, amsgrad=cfg.SOLVER.AMSGRAD
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.SPARSE_INST.DATASET_MAPPER == "SparseInstDatasetMapper":
            from sparseinst import SparseInstDatasetMapper
            mapper = SparseInstDatasetMapper(cfg, is_train=True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(0, IterationHook())
        return hooks


def configure_trainer(train):
    # basic setup
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    config_path = train.model_info["meta"]["model_files"]["config"]
    cfg.merge_from_file(config_path)
    
    # weights path
    checkpoint_path = train.model_info["meta"]["model_files"]["checkpoint"]
    if sly.is_development():
        checkpoint_path = "." + checkpoint_path
    cfg.MODEL.WEIGHTS = checkpoint_path

    # configure datasets
    cfg.DATASETS.TRAIN = ("main_train",)
    cfg.DATASETS.TEST = ("main_validation",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train.classes)

    # set output dir
    output_dir = "./train_output"
    if os.path.exists(output_dir):
        sly.fs.clean_dir(output_dir)
    else:
        os.mkdir(output_dir)
    cfg.OUTPUT_DIR = output_dir

    # set training parameters
    hyperparameters = train.hyperparameters
    cfg.SOLVER.MAX_ITER = hyperparameters["n_iterations"]
    cfg.SOLVER.IMS_PER_BATCH = hyperparameters["batch_size"]
    cfg.SOLVER.BASE_LR = hyperparameters["base_lr"]
    cfg.SOLVER.STEPS = tuple(range(hyperparameters["lr_decay_frequency"], cfg.SOLVER.MAX_ITER, hyperparameters["lr_decay_frequency"]))
    cfg.SOLVER.GAMMA = hyperparameters["lr_decay_factor"]
    cfg.SOLVER.OPTIMIZER = hyperparameters["optimizer"].upper()
    cfg.DATALOADER.NUM_WORKERS = hyperparameters["n_workers"]
    cfg.SOLVER.CHECKPOINT_PERIOD = hyperparameters["checkpoint_save_frequency"]
    cfg.TEST.EVAL_PERIOD = hyperparameters["evaluation_frequency"]
    cfg.SOLVER.AMP.ENABLED = hyperparameters["use_fp16"]
    cfg.freeze()

    # save config
    with open("./train_output/training_config.yaml", "w") as config_file:
        config_file.write(cfg.dump())

    # initialize trainer
    trainer = Trainer(cfg, sly_logger=train_logger)
    trainer.resume_or_load(resume=False)
    return trainer



