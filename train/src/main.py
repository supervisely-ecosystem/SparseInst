from train.src.custom_train_app import CustomTrainApp
from dotenv import load_dotenv
from train.src.data_converter import convert_supervisely_to_segmentation, configure_datasets
from train.src.model_configuration import configure_trainer
from serve.src.sparseinst_model import SparseInstModel
import supervisely as sly
import os
import torch
from serve.src.onnx_utils import convert_to_onnx
from serve.src.tensorrt_utils import convert_to_tensorrt
from detectron2.config import get_cfg
from sparseinst import add_sparse_inst_config


load_dotenv("supervisely.env")
load_dotenv("local.env")


train = CustomTrainApp(
    framework_name="SparseInst",
    models="models/models.json",
    hyperparameters="train/src/hyperparameters.yaml",
    app_options="train/src/app_options.yaml",
    work_dir="train_data",
)
train.register_inference_class(SparseInstModel)

def clean_data():
    # delete app data since it is no longer needed
    sly.fs.remove_dir("train_data")
    sly.fs.remove_dir("train_output")

train.app.call_before_shutdown(clean_data)


@train.start
def start_training():
    # prepare training dataset
    project_seg_dir = convert_supervisely_to_segmentation(train)
    configure_datasets(train, project_seg_dir)
    # prepare model
    trainer = configure_trainer(train)
    # train model
    output_dir = "./train_output"
    train.start_tensorboard(output_dir)
    trainer.train()
    # save class names to checkpoint
    best_checkpoint_path = os.path.join(output_dir, "model_final.pth")
    best_checkpoint_dict = torch.load(best_checkpoint_path)
    best_checkpoint_dict["class_names"] = train.classes
    torch.save(best_checkpoint_dict, best_checkpoint_path)
    # generate experiment info
    config_path = os.path.join(output_dir, "training_config.yaml")
    experiment_info = {
        "model_name": train.model_name,
        "model_files": {"config": config_path},
        "checkpoints": output_dir,
        "best_checkpoint": "model_final.pth",
        "task_type": "instance segmentation",
    }
    return experiment_info


@train.export_onnx
def to_onnx(experiment_info: dict):
    config_path = experiment_info["model_files"]["config"]
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(config_path)

    checkpoint_path = experiment_info["best_checkpoint"]
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = 0.1

    onnx_path = convert_to_onnx(cfg)
    return onnx_path


@train.export_tensorrt
def to_tensorrt(experiment_info: dict):
    config_path = experiment_info["model_files"]["config"]
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(config_path)

    checkpoint_path = experiment_info["best_checkpoint"]
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = 0.1

    onnx_path = convert_to_onnx(cfg)
    tensorrt_path = convert_to_tensorrt(onnx_path)
    return tensorrt_path