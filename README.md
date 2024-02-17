# UVM-Net
U-shaped Vision Mamba for Single Image Dehazing
![image](https://github.com/zzr-idam/UVM-Net/blob/main/fw3.png)

## Abstract
Currently, Transformer is the most popular architecture for image dehazing, but due to its large computational complexity, its ability to handle long-range dependency is limited on resource-constrained devices. To tackle this challenge, we introduce the U-shaped Vision Mamba (UVM-Net), an efficient single-image dehazing network. Inspired by the State Space Sequence Models (SSMs), a new deep sequence model known for its power to handle long sequences, we design a Bi-SSM block that integrates the local feature extraction ability of the convolutional layer with the ability of the SSM to capture long-range dependencies. Extensive experimental results demonstrate the effectiveness of our method. Our method provides a more highly efficient idea of long-range dependency modeling for image dehazing as well as other image restoration tasks.

## Installation

- `pip install causal-conv1d>=1.1.0`: an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
- `pip install mamba-ssm`: the core Mamba package.

It can also be built from source with `pip install .` from this repository.

If `pip` complains about PyTorch versions, try passing `--no-build-isolation` to `pip`.

Other requirements:
- Linux
- NVIDIA GPU (48G RAM)
- PyTorch 1.12+
- CUDA 11.6+

## Download

You can download the datasets on [GoogleDrive](https://drive.google.com/drive/folders/1Yy_GH6_bydYPU6_JJzFQwig4LTh86VI4?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1WVdNccqDMnJ5k5Q__Y2dsg?pwd=gtuw) (gtuw).

## UVMB
We show the basic modules of our network, with continuous improvements to follow.

## Train
```
python Train.py
```

## Demo
![image](https://github.com/zzr-idam/UVM-Net/blob/main/demo.jpg)


## Citation
To the best of our knowledge is the first image enhancement method to introduce Mamba technique in the low-level domain.
If you find this work useful for your research, please cite our paper:
```
@article{zheng2024u,
  title={U-shaped Vision Mamba for Single Image Dehazing},
  author={Zheng, Zhuoran and Wu, Chen},
  journal={arXiv preprint arXiv:2402.04139},
  year={2024}
}
```
