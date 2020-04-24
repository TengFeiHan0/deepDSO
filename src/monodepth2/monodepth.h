#ifndef MONODEPTH_H_
#define MONODEPTH_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;
using namespace cv;
using namespace at;

namespace dso{


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
            cv::Mat inference(cv::Mat &images,const  int &height, const int &width);
            
            
            std::string model_file_; // the path of the given model
            bool use_gpu_= true; // use GPU or CPU
            
            torch::jit::script::Module model;
    };     

} // namespace monodepth

#endif