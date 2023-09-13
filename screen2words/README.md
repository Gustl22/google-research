# Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning
This repository contains the model code and the experimental framework for "Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning" by Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman and Yang Li, which is conditionally accepted in UIST 2021.

# Data
The 112,085 screen summaries we collected for model training and evaluation can be downloaded from https://github.com/google-research-datasets/screen2words

# Implementation
The screen2words models are implemented based on Transformer implementation in [TensorFlow Model Garden](https://github.com/tensorflow/models).

# Nvidia CUDA installation

Visit Nvidia [Cuda Archive](https://developer.nvidia.com/cuda-toolkit-archive).
Select the version mentioned in [Tensorflow installation](https://www.tensorflow.org/install/pip). Select e.g. the deb (local), as otherwise the newest Cuda will be installed.
Follow the installation instructions, but install with `sudo apt-get install cuda-11.8`
Validate with `nvcc --version`

Execute: `python3 -m pip install nvidia-cudnn-cu11==8.6.0.163`
Validate CUDA is available with:

Add environment variables:
```shell
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn>
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH'
```

```shell
python3
>>> import torch
>>> torch.cuda.is_available()
True
```
