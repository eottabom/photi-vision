FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-pip python3-tk wget curl git vim libsm6 libxext6 libxrender-dev screen libopencv-dev
RUN ln -sf /usr/bin/python3 /usr/bin/python && python -V && which python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip && pip -V && which pip
RUN pip3 install -U pip six numpy wheel setuptools mock future>=0.17.1 opencv-python matplotlib pillow scikit-image

RUN mkdir /workspace
WORKDIR /workspace
ADD ./source /workspace/
RUN make
#ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
