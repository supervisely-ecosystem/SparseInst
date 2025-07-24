FROM supervisely/base-py-sdk:6.73.404

RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip3 install setuptools==69.5.1
RUN pip3 install imagecorruptions==1.1.2
RUN pip3 install ffmpeg-python==0.2.0
RUN pip3 install protobuf==3.20.3

RUN pip3 install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

RUN apt-get -y install nvidia-cuda-dev
RUN pip3 install pycuda==2025.1.1
RUN pip3 install nvidia-tensorrt==99.0.0

RUN pip3 install Pillow==9.5.0
RUN pip3 install timm==1.0.16
RUN pip3 install onnxruntime-gpu==1.11.0
RUN pip3 install supervisely[model-benchmark]==6.73.404
RUN pip3 install supervisely[training]==6.73.404

RUN apt-get update && apt-get -y install curl

RUN mkdir -p /weights

RUN curl -L -o /weights/sparse_inst_r50_giam_ceaffc.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_r50_giam_ceaffc.pth"
RUN curl -L -o /weights/sparse_inst_r50vd_base.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_r50vd_base.pth"
RUN curl -L -o /weights/sparse_inst_r50_dcn_giam_aug.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_r50_dcn_giam_aug.pth"
RUN curl -L -o /weights/sparse_inst_r50vd_dcn_giam_aug.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_r50vd_dcn_giam_aug.pth"
RUN curl -L -o /weights/sparse_inst_pvt_v2_b2_li_giam.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_pvt_v2_b2_li_giam.pth"

LABEL python_sdk_version=6.73.404