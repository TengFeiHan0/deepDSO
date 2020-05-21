# deepDSO

# Abstract
In this project, we extend direct sparse odometry(DSO) with a self-supervised depth estimation module named [packnet-sfm](https://github.com/TRI-ML/packnet-sfm). 

|Sequence on KITTI|DVSO|deepDSO|DSO|ORB-SLAM|
|---|---|---|---|---:|
|00|**0.71**|15.13|188 |77.95|
|01|**1.18**|5.901|9.17|X|
|02|**0.84**|12.53|114|41.00|
|03|**0.79**|1.516|17.7|**1.018**|
|04|0.35|**0.100**|0.82|0.930|
|05|**0.58**|20.3|72.6|40.35|
|06|**0.71**|1.547|42.2|52.22|
|07|**0.73**|8.369|48.4|16.54|
|08|**1.03**|10.53|177|51.62|
|09|**0.83**|14.00|28.1|58.17|
|10|**0.84**|4.10|24.0|18.47|
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
