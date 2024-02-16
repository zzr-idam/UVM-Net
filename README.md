# UVM-Net
U-shaped Vision Mamba for Single Image Dehazing
![image](https://github.com/zzr-idam/UVM-Net/blob/main/fw3.png)


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

# UVMB
We show the basic modules of our network, with continuous improvements to follow.

# Demo
![image](https://github.com/zzr-idam/UVM-Net/blob/main/demo.jpg)


## Citation

If you use this codebase, or otherwise found our work valuable, please cite Mamba:
```
@article{zheng2024u,
  title={U-shaped Vision Mamba for Single Image Dehazing},
  author={Zheng, Zhuoran and Wu, Chen},
  journal={arXiv preprint arXiv:2402.04139},
  year={2024}
}
```
