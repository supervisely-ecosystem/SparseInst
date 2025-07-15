import torch
from torch import nn
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import os
import cv2
import numpy as np
from detectron2.structures import Instances


def convert_to_onnx(cfg):
    cfg.MODEL.META_ARCHITECTURE = "SparseInst_ONNX_TRT"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.NORM = "BN"
    cfg.freeze()

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    checkpointer = DetectionCheckpointer(model)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)

    height, width = 640, 640
    dummy_input = torch.rand((3, height, width)).unsqueeze(0).to(cfg.MODEL.DEVICE)

    input_names = ["images"]
    output_names = ["scores", "classes", "masks"]

    model.forward = model.forward_test_3
    model.eval()

    if not os.path.exists("output"):
        os.mkdir("output")
    
    onnx_model_path = "output/sparseinst.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=False,
        opset_version=11,
    )
    return onnx_model_path


def preprocess_onnx_inputs(images_np, cfg):
    images = []
    orig_shapes = []
    for image in images_np:
        orig_shapes.append({"height": image.shape[0], "width": image.shape[1]})
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32)
        mean = np.array(cfg.MODEL.PIXEL_MEAN)
        std = np.array(cfg.MODEL.PIXEL_STD)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        images.append(image)
    return np.asarray(images).astype(np.float32), orig_shapes


def postprocess_onnx_predictions(scores, classes, masks, orig_shapes, cfg):
    mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
    scores = scores[0]
    classes = classes[0]
    results = []
    for orig_shape, scores_per_image, classes_per_image, masks_per_image in zip(orig_shapes, scores, classes, masks):
        height, width = orig_shape["height"], orig_shape["width"]
        mask_pred_per_image = masks_per_image.reshape((100, 640//4, 640//4))
        m = nn.UpsamplingBilinear2d(size=(height, width))
        mask_pred_per_image  = torch.tensor(mask_pred_per_image)
        mask_pred = m(mask_pred_per_image.unsqueeze(0)).squeeze(0)
        mask_pred = mask_pred > mask_threshold

        predictions_mask = mask_pred.reshape((100,height,width))
        ori_shape = (1, 3, height, width)
        result = Instances(ori_shape)
        result.pred_masks = predictions_mask
        result.scores = scores_per_image
        result.pred_classes = classes_per_image
        results.append(result)
    return results



