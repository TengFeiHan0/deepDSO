#ifndef MONODEPTH_H_
#define MONODEPTH_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
using namespace std;
using namespace cv;
using namespace at;
namespace monodepth2{


typedef enum _DEVICE_TYPE_
{
   CPU = -1,
   GPU = 1
}DEVICE_TYPE;

class MonoDepth{
        public:
            MonoDepth(const std::string &model_file, int use_gpu);

            ~MonoDepth();

            /// Infer depth from image (implementation)
            void inference(cv::Mat& image, cv::Mat& depth);
            //transform disp into depth
            void disp2Depth(cv::Mat &dispMap, cv::Mat &depthMap);
        private:

            // inference depth from inputs
            cv::Mat inference(cv::Mat &images, int height, int width);
            
            
            std::string model_file_; // the path of the given model
            int use_gpu_; // use GPU or CPU
            torch::jit::script::Module model;
    };     

} // namespace monodepth

#endif