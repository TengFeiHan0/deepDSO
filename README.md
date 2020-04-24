# deepDSO(under construction)

# Abstract
In this project, we extend direct sparse odometry(DSO) with a self-supervised depth estimation module named [packnet-sfm](https://github.com/TRI-ML/packnet-sfm). 

# Installation
## Dependencies
### required
- [libtorch](https://pytorch.org/get-started/locally/)
- OpenCV3+
- CUDA(10.1)
- [Protobuf >= 3.8.x](https://github.com/google/protobuf/releases)
### optional
TensorRT is one the applicable inference framework which reduces the consumed inference time and computation overhead greatly. Although converting model into torchscript is also a good choice to save time, the builtin optimization principles of tensorrt are still much effcient than torchscript. I also trying to optimize this depth estimaition module with TensorRT.
- ONNX(1.6.0)
- [TensorRT 7.0 open source libaries (master branch)](https://github.com/NVIDIA/TensorRT/)

## Building
- prepare all required libs mentioned before, and download this projet.

             git clone https://github.com/TengFeiHan0/deepDSO.git 
- refer to [monodepth2.cpp](https://github.com/TengFeiHan0/monodepth2.cpp) and download the required torchscript model(`packnet.pt`), please don't forget to download `libtorch` and put this dir under `deepDSO/lib/` 
- Build

		cd deepDSO
		mkdir build
		cd build
		cmake ..
		make -j4
