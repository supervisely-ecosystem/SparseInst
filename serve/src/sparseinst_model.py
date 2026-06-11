import os
import warnings
warnings.filterwarnings("ignore")
from detectron2.config import get_cfg
from sparseinst import add_sparse_inst_config
import supervisely as sly
from detectron2.data import MetadataCatalog
from supervisely.nn.inference import CheckpointInfo, ModelSource, RuntimeType, Timer
import numpy as np
from typing import List
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import torch
from detectron2.structures import ImageList
from sparseinst import SparseInst
from sparseinst.utils import nested_tensor_from_tensor_list
from serve.src.onnx_utils import convert_to_onnx, preprocess_onnx_inputs, postprocess_onnx_predictions
import onnxruntime as ort
from serve.src.tensorrt_utils import TRTInference, convert_to_tensorrt
import yaml



@torch.jit.script
def normalizer(x, mean, std):
    return (x - mean) / std


torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = False


class SparseInstModel(sly.nn.inference.InstanceSegmentation):
    FRAMEWORK_NAME = "SparseInst"
    MODELS = "models/models.json"
    APP_OPTIONS = "serve/src/app_options.yaml"
    INFERENCE_SETTINGS = "serve/src/inference_settings.yaml"

    def load_model(
        self,
        model_files: dict,
        model_info: dict,
        model_source: str,
        device: str,
        runtime: str,
    ):
        config_path = model_files["config"]
        checkpoint_path = model_files["checkpoint"]
        # if sly.is_development():
        #     checkpoint_path = "." + checkpoint_path
        self.cfg = self._build_cfg(config_path, checkpoint_path, device)
        self.device = device

        if runtime == RuntimeType.PYTORCH:
            self.cfg.freeze()
            self.model = SparseInst(self.cfg)
            self.model.to(self.device)
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(self.cfg.MODEL.WEIGHTS)
            self.model.eval()
        elif runtime in [RuntimeType.ONNXRUNTIME, RuntimeType.TENSORRT]:
            checkpoint_ext = os.path.splitext(checkpoint_path)[1].lower()

            if runtime == RuntimeType.ONNXRUNTIME:
                if checkpoint_ext == ".onnx":
                    onnx_path = checkpoint_path
                else:
                    onnx_path, _ = self._get_export_paths(checkpoint_path)
                    onnx_path = convert_to_onnx(self.cfg, onnx_path)
                providers = (["CUDAExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"])
                self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
            elif runtime == RuntimeType.TENSORRT:
                if checkpoint_ext == ".engine":
                    tensorrt_path = checkpoint_path
                else:
                    onnx_path, tensorrt_path = self._get_export_paths(checkpoint_path)
                    if checkpoint_ext == ".onnx":
                        onnx_path = checkpoint_path
                    else:
                        onnx_path = convert_to_onnx(self.cfg, onnx_path)
                    tensorrt_path = convert_to_tensorrt(onnx_path, tensorrt_path)
                self.engine = TRTInference(tensorrt_path, max_batch_size=1)

        if model_source == ModelSource.PRETRAINED:
            self.checkpoint_info = CheckpointInfo(
                checkpoint_name=os.path.basename(checkpoint_path),
                model_name=model_info["meta"]["model_name"],
                architecture=self.FRAMEWORK_NAME,
                checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
                model_source=model_source,
            )
            self.classes = MetadataCatalog.get("coco_2017_val").thing_classes
        else:
            self.classes = self._load_custom_classes(checkpoint_path)

        obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in self.classes]
        conf_tag = sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
        self._model_meta = sly.ProjectMeta(obj_classes=obj_classes, tag_metas=[conf_tag])

        self.runtime = runtime
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        with open(self.INFERENCE_SETTINGS) as settings_file:
            self._custom_inference_settings = yaml.safe_load(settings_file)

    def predict_benchmark(self, images_np: List[np.ndarray], settings: dict = None):
        if self.runtime == RuntimeType.PYTORCH:
            return self._predict_pytorch(images_np, settings)
        elif self.runtime == RuntimeType.ONNXRUNTIME:
            return self._predict_onnx(images_np, settings)
        elif self.runtime == RuntimeType.TENSORRT:
            return self._predict_tensorrt(images_np, settings)

    def _predict_pytorch(
        self, images_np: List[np.ndarray], settings: dict = None
    ):
        # 1. Preprocess
        with Timer() as preprocess_timer:
            images, orig_shapes = self._preprocess_input(images_np)
            max_shape = images.tensor.shape[2:]
        # 2. Inference
        with Timer() as inference_timer:
            fp16 = settings.get("fp16", False)
            with torch.cuda.amp.autocast(enabled=fp16):
                with torch.no_grad():
                    features = self.model.backbone(images.tensor)
                    features = self.model.encoder(features)
                    output = self.model.decoder(features)
        # 3. Postprocess
        with Timer() as postprocess_timer:
            results = self.model.inference(output, orig_shapes, max_shape, images.image_sizes)
            predictions = self._format_predictions(results, settings)
            
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark
    
    def _predict_onnx(
        self, images_np: List[np.ndarray], settings: dict = None
    ):
        # 1. Preprocess
        with Timer() as preprocess_timer:
            images, orig_shapes = preprocess_onnx_inputs(images_np, self.cfg)
        # 2. Inference
        with Timer() as inference_timer:
            scores, classes, masks = self.ort_session.run(
                output_names=None,
                input_feed={"images": images},
            )
        # 3. Postprocess
        with Timer() as postprocess_timer:
            results = postprocess_onnx_predictions([scores], [classes], [masks], orig_shapes, self.cfg)
            predictions = self._format_predictions(results, settings)
            
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark
    
    def _predict_tensorrt(
        self, images_np: List[np.ndarray], settings: dict = None
    ):
        # 1. Preprocess
        with Timer() as preprocess_timer:
            images, orig_shapes = preprocess_onnx_inputs(images_np, self.cfg)
            images = torch.from_numpy(images).to(self.device)
        # 2. Inference
        with Timer() as inference_timer:
            output = self.engine({"images": images})
            scores = output["scores"].cpu()
            classes = output["classes"].cpu()
            masks = output["masks"].cpu()
        # 3. Postprocess
        with Timer() as postprocess_timer:
            results = postprocess_onnx_predictions([scores], [classes], [masks], orig_shapes, self.cfg)
            predictions = self._format_predictions(results, settings)
            
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark
        

    def _preprocess_input(self, images_np: List[np.ndarray]):
        orig_shapes = [{"height": img.shape[0], "width": img.shape[1]} for img in images_np]
        images = [self.aug.get_transform(img).apply_image(img) for img in images_np]
        images = [torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) for img in images]
        images = [img.to(self.device) for img in images]
        if self.runtime == RuntimeType.PYTORCH:
            images = [self.model.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, 32)
        elif self.runtime == RuntimeType.TENSORRT:
            self.pixel_mean = torch.Tensor(
                self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
            self.pixel_std = torch.Tensor(
                self.cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
            
            def normalizer(image):
                image = (image - self.pixel_mean) / self.pixel_std
                return image
        
            images = [normalizer(x) for x in images]

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        return images, orig_shapes
    
    def _format_prediction(self, result, confidence_threshold):
        prediction_figures = []
        if self.runtime ==  RuntimeType.PYTORCH:
            pred_classes = result.pred_classes.detach().cpu().numpy()
            valid = np.where(pred_classes <= len(self.classes) - 1)[0]
            pred_classes = pred_classes[valid]
            pred_scores = result.scores.detach().cpu().numpy()
            pred_scores = pred_scores[valid].tolist()
            pred_masks = result.pred_masks.detach().cpu().numpy()
            pred_masks = pred_masks[valid]
        else:
            pred_classes = result.pred_classes
            pred_scores = result.scores.tolist()
            pred_masks = result.pred_masks

        pred_class_names = [self.classes[pred_class] for pred_class in pred_classes]

        for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
            if score >= confidence_threshold and mask.any():
                prediction_figures.append(sly.nn.PredictionMask(class_name, mask, score))
        return prediction_figures

    def _format_predictions(self, results, settings):
        confidence_threshold = settings.get("conf_thresh", 0.5)
        predictions = []
        for result in results:
            if len(result.scores) > 0:
                prediction = self._format_prediction(result, confidence_threshold)
                predictions.append(prediction)
        return predictions

    def _build_cfg(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        cfg = get_cfg()
        add_sparse_inst_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = checkpoint_path
        cfg.MODEL.DEVICE = device
        cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = 0.1
        return cfg

    def _get_export_paths(self, checkpoint_path: str):
        checkpoint_root, _ = os.path.splitext(checkpoint_path)
        return f"{checkpoint_root}.onnx", f"{checkpoint_root}.engine"

    def _load_custom_classes(self, checkpoint_path: str) -> List[str]:
        checkpoint_root, checkpoint_ext = os.path.splitext(checkpoint_path)
        checkpoint_candidates = [checkpoint_path]
        if checkpoint_ext.lower() not in (".pt", ".pth"):
            checkpoint_candidates = [f"{checkpoint_root}.pth", f"{checkpoint_root}.pt"]

        for candidate in checkpoint_candidates:
            if os.path.exists(candidate):
                return torch.load(candidate)["class_names"]

        raise FileNotFoundError(
            "Could not load custom class names. Expected a PyTorch checkpoint "
            f"near exported file '{checkpoint_path}'."
        )

    # Converters --------------- #
    def export_onnx(self, deploy_params: dict) -> str:
        model_files = deploy_params["model_files"]
        checkpoint_path = model_files["checkpoint"]
        config_path = model_files["config"]
        device = deploy_params.get("device", "cuda")
        cfg = self._build_cfg(config_path, checkpoint_path, device)
        onnx_path, _ = self._get_export_paths(checkpoint_path)
        return convert_to_onnx(cfg, onnx_path)

    def export_tensorrt(self, deploy_params: dict) -> str:
        model_files = deploy_params["model_files"]
        checkpoint_path = model_files["checkpoint"]
        config_path = model_files["config"]
        device = deploy_params.get("device", "cuda")
        onnx_path, tensorrt_path = self._get_export_paths(checkpoint_path)

        if not os.path.exists(onnx_path):
            cfg = self._build_cfg(config_path, checkpoint_path, device)
            convert_to_onnx(cfg, onnx_path)

        return convert_to_tensorrt(onnx_path, tensorrt_path)
