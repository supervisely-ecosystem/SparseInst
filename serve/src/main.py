import os
from dotenv import load_dotenv
import supervisely as sly
from serve.src.sparseinst_model import SparseInstModel

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv("supervisely.env")

model = SparseInstModel(
    use_gui=True,
    use_serving_gui_template=True,
)
model.serve()