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

RUN apt-get update && apt-get -y install wget
RUN apt-get -y install curl
RUN apt-get update && apt -y install wireguard iproute2

RUN pip3 install Pillow==9.5.0
RUN pip3 install timm==1.0.16
RUN pip3 install gdown==5.2.0
RUN pip3 install onnxruntime-gpu==1.11.0
RUN pip3 install supervisely[model-benchmark]==6.73.404
RUN pip3 install supervisely[training]==6.73.404

LABEL "role"="development"