#include "monodepth.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
int main(int argc, char** argv){

    const int image_height = 192;
    const int image_width = 640;
    const string ts_model = "/home/fei/LDSO/vocab/packnet.pt";
    int use_gpu = 1;
    monodepth::MonoDepth md(ts_model, use_gpu);
    std::string img_path = "/home/fei/cityscapes/test_image.png";
    cv::Mat depth;
    cv::Mat img(cv::imread(img_path));

    md.inference(img, depth, image_height, image_width);

    cv::namedWindow("MONODEPTH2", CV_MINOR_VERSION);
    cv::imshow("MONODEPTH2", depth);
    cv::waitKey(10000);

}