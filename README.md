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
- NVIDIA GPU (12G RAM)
- PyTorch 1.12+
- CUDA 11.6+

## Download

You can download the datasets on [GoogleDrive](https://drive.google.com/drive/folders/1Yy_GH6_bydYPU6_JJzFQwig4LTh86VI4?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1WVdNccqDMnJ5k5Q__Y2dsg?pwd=gtuw) (gtuw).

## UVMB
We show the basic modules of our network, with continuous improvements to follow.

There is no need to reduce the number of layers in the UNet, my basic module has three SSMs, you can remove the one that is the most computationally intensive (after reshaping); besides, you can conduct a ReLU or Sigmoid on the output of the UNet and do a residual or multiplication operation with the original input image.

output = nn.ReLU()(UNet(x))

output = output * x - output + 1 // AODNet equation

return output

## Train
```
python Train.py
```

## The Potential of Models
Our method can perform tasks such as image dehazingg, deraining, and low-light enhancement.
The first three images are the test images, the middle three images are the output of the model, and the last three are the GT.

![image](https://github.com/zzr-idam/UVM-Net/blob/main/demo.jpg)

![image](https://github.com/zzr-idam/UVM-Net/blob/main/low-light.jpg)

![image](https://github.com/zzr-idam/UVM-Net/blob/main/deblur.jpg)

![image](https://github.com/zzr-idam/UVM-Net/blob/main/derain.jpg)

![image](https://github.com/zzr-idam/UVM-Net/blob/main/underwater.jpg)

## Note
Our method can also be performed on UHD (4K resolution) images on a single GPU, you just need to do the following:
```
 def forward(self, x):
        x1 = F.interpolate(x, size=[512, 512], mode='bilinear', align_corners=True)
        x1 = our_network(x1)    
        output = F.interpolate(x1, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True) + x
        return output
```

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
