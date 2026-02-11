FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update

RUN apt-get install -y python3.12 python3.12-dev wget git build-essential
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py
RUN pip install --upgrade setuptools

RUN pip3 install setuptools==69.5.1
RUN pip3 install imagecorruptions==1.1.2
RUN pip3 install ffmpeg-python==0.2.0
RUN pip3 install protobuf==3.20.3

RUN pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install ninja
RUN python3.12 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN pip3 install pycuda>=2024.1
RUN pip3 install nvidia-tensorrt==10.0.1

RUN pip3 install Pillow==9.5.0
RUN pip3 install timm==1.0.16
RUN pip3 install onnxruntime-gpu==1.19.0

RUN apt-get update && apt-get -y install curl

RUN mkdir -p /weights

RUN curl -L -o /weights/sparse_inst_r50_giam_ceaffc.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_r50_giam_ceaffc.pth"
RUN curl -L -o /weights/sparse_inst_r50vd_base.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_r50vd_base.pth"
RUN curl -L -o /weights/sparse_inst_r50_dcn_giam_aug.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_r50_dcn_giam_aug.pth"
RUN curl -L -o /weights/sparse_inst_r50vd_dcn_giam_aug.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_r50vd_dcn_giam_aug.pth"
RUN curl -L -o /weights/sparse_inst_pvt_v2_b2_li_giam.pth "https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparse_inst_pvt_v2_b2_li_giam.pth"

RUN pip3 install supervisely==6.73.530
RUN pip3 install supervisely[model-benchmark]==6.73.530
RUN pip3 install supervisely[training]==6.73.530

LABEL python_sdk_version="6.73.530"
