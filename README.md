# deepDSO

# Abstract
In this project, we extend direct sparse odometry(DSO) with a self-supervised depth estimation module named [packnet-sfm](https://github.com/TRI-ML/packnet-sfm). 

|Sequence on KITTI|DVSO|CNN-DSO|DSO|ORB-SLAM|
|---|---|---|---|---:|
|00||**15.13**|113.18|77.95|
|01||**5.901**|X|X|
|02||**12.53**|116.81|41.00|
|03||1.516|1.3943|**1.018**|
|04||**0.100**|0.422|0.930|
|05||**20.3**|47.46|40.35|
|06||**1.547**|55.61|52.22|
|07||**8.369**|16.71|16.54|
|08||**10.53**|111.08|51.62|
|09||**14.00**|52.22|58.17|
|10||**4.10**|11.09|18.47|
# Installation
## Dependencies
### required
- [libtorch](https://pytorch.org/get-started/locally/)(`c++11`)
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
- go to [monodepth2.cpp](https://github.com/TengFeiHan0/monodepth2.cpp) and download the required torchscript model(`packnet.pt`) and another required lib (`libtorch`) from its offical [website](https://pytorch.org/get-started/locally/)
- Build

		cd deepDSO
		mkdir build
		cd build
		cmake ..
		make -j4
