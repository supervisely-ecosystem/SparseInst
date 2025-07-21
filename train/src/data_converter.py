import supervisely as sly
import os
import json
from PIL import Image
import functools
from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np
import pycocotools.mask
from detectron2.structures import BoxMode


def convert_supervisely_to_segmentation(train):
    project_dir_seg = train.project_dir + "_seg"

    if sly.fs.dir_exists(project_dir_seg) is False:  # for debug, has no effect in production
        sly.fs.mkdir(project_dir_seg, remove_content_if_exists=True)
        sly.Project.to_segmentation_task(
            train.project_dir,
            project_dir_seg,
            target_classes=train.classes,
            progress_cb=None,
            segmentation_type="instance",
        )

    return project_dir_seg


def get_image_info(image_path):
    im = Image.open(image_path)
    width, height = im.size

    return width, height


def string2number(s):
    return int.from_bytes(s.encode(), 'little')


def mask_to_image_size(label, existence_mask, img_size):
    mask_in_images_coordinates = np.zeros(img_size, dtype=bool)  # size is (h, w)

    row, column = label.geometry.origin.row, label.geometry.origin.col  # move mask to image space
    mask_in_images_coordinates[row: row + existence_mask.shape[0], column: column + existence_mask.shape[1]] = \
        existence_mask

    return mask_in_images_coordinates


def get_objects_on_image(ann, all_classes):
    objects_on_image = []

    for label in ann.labels:
        rect = label.geometry.to_bbox()

        seg_mask = np.asarray(label.geometry.convert(sly.Bitmap)[0].data).copy()
        seg_mask_in_image_coords = np.asarray(mask_to_image_size(label, seg_mask, ann.img_size))

        rle_seg_mask = pycocotools.mask.encode(np.asarray(seg_mask_in_image_coords, order="F"))

        obj = {
            "bbox": [rect.left, rect.top, rect.right, rect.bottom],
            "bbox_mode": BoxMode.XYXY_ABS,
            # "segmentation": [new_poly],
            "segmentation": rle_seg_mask,
            "category_id": all_classes[label.obj_class.name],
        }

        objects_on_image.append(obj)

    return objects_on_image


def convert_data_to_detectron(dataset, project_meta, all_classes):
    records = []
    for item_name, image_path, ann_path in dataset.items():
        record = {
            "file_name": image_path,
            "image_id": string2number(item_name)
        }
        width, height = get_image_info(record["file_name"])
        record["height"] = height
        record["width"] = width
        ann = sly.Annotation.load_json_file(ann_path, project_meta)
        record["annotations"] = get_objects_on_image(ann, all_classes)
        records.append(record)
    return records


def configure_datasets(train, project_seg_dir_path):
    project = sly.Project(directory=project_seg_dir_path, mode=sly.OpenMode.READ)
    project_meta = project.meta

    all_classes = {}
    for class_index, obj_class in enumerate(project_meta.obj_classes):
        all_classes[obj_class.name] = class_index

    project_path = os.path.join(train.work_dir, "sly_project")
    train_ds_path = os.path.join(project_path, "train")
    val_ds_path = os.path.join(project_path, "val")

    train_ds, val_ds = sly.Dataset(train_ds_path, sly.OpenMode.READ), sly.Dataset(val_ds_path, sly.OpenMode.READ)
    
    get_train = functools.partial(convert_data_to_detectron, train_ds, project_meta, all_classes)
    get_validation = functools.partial(convert_data_to_detectron, val_ds, project_meta, all_classes)
    
    DatasetCatalog.register("main_train", get_train)
    DatasetCatalog.register("main_validation", get_validation)

    MetadataCatalog.get("main_train").thing_classes = train.classes
    MetadataCatalog.get("main_validation").thing_classes = train.classes
    MetadataCatalog.get("main_validation").evaluator_type = "coco"