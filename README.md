# deepDSO

# Abstract
In this project, we extend direct sparse odometry(DSO) with a self-supervised depth estimation module named [packnet-sfm](https://github.com/TRI-ML/packnet-sfm). 


|Sequence on KITTI|00|01|02|03|04|05|06|07|08|09|10|
|---|---|---|---|---|---|---|---|---|---|---|---:|
|ORBSLAM|77.95|X|41.0|1.0118|0.930|40.35|52.22|16.54|51.62|58.17|18.47|
|DSO|188|9.17|114|17.7|0.82|72.6|42.2|48.4|177|28.1|24.0|
|DVSO|**0.71**|**1.18**|**0.84**|**0.79**|0.35|**0.58**|**0.71**|**0.73**|**1.03**|**0.83**|**0.84**|
|My|15.13|5.901|12.53|1.516|**0.100**|20.3|1.547|8.369|10.53|14.00|4.10|
|D3VO |0 | 1.07|2|3|4|5|6|7|8|9|10|


| Models  | Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|---------|---------|--------|-------|-----------|-------|-------|-------|
| monodepth2[1] | 0.115 | 0.903 | 4.863 | 0.193 | 0.877 | 0.959 | 0.981 |
| packnet-sfm[2] | 0.111 | 0.785 | 4.601 | 0.189 | 0.878 | 0.960 | 0.982 |
| packnet-semantic[3] | 0.100 | 0.761 | 4.270 | 0.175 | **0.902** | **0.965** | 0.982 |
| DVSO[4] | **0.092** | **0.547** | **3.390** | **0.177** | 0.898 | 0.962 | 0.982 |
| our | 0.113 | 0.818 | 4.621 | 0.190 | 0.875 | 0.958 |0.982 |
|D3VO | 0.099 | 0.763 | 4.485 |0.185  |0.763  | 0.885 | 0.958 |0.979 |
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
