#include "monodepth.h"
#include "my_utils.h"

//headers for pytorch(c++)
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <assert.h>

using namespace std;
using namespace cv; // opencv
using namespace at; //pytorch c++ api

namespace monodepth
{
  //read torchscript
MonoDepth::MonoDepth(const std::string &model_file, int use_gpu) : model_file_(model_file), use_gpu_(use_gpu)
{

    model = torch::jit::load(model_file);
    if (DEVICE_TYPE::GPU == use_gpu)
    {
        model.to(at::kCUDA);
    }
    else
    {
        model.to(at::kCPU);
    }
}


// Destructor (interface)
MonoDepth::~MonoDepth() = default;

void MonoDepth::inference(cv::Mat &image, cv::Mat &depth, int height, int width){
    depth = MonoDepth::inference(image, height, width);
    //MonoDepth::disp2Depth(disp, depth);
    cout<<"got final depthmap"<<endl;
}

cv::Mat MonoDepth::inference(cv::Mat &image, int height, int width)
{
    //images_to_tensors
    assert(!images.empty());
    cv::Mat input_mat;
    cv::resize(image, input_mat, cv::Size(width, height));
    //[0, 255]
    input_mat.convertTo(input_mat,CV_32FC1, 1./255.);
    //[0, 1]
    //transform a cv::Mat into a tensor
    torch::Tensor tensor_image = torch::from_blob(input_mat.data, {1,input_mat.rows, input_mat.cols,3}, torch::kF32);
    tensor_image = tensor_image.permute({0,3,1,2});

    if (use_gpu_ == 1)
    {
        tensor_image = tensor_image.to(at::kCUDA);
    }
    else
    {
        tensor_image = tensor_image.to(at::kCPU);
    }
    //[0,1]
    vector<torch::IValue> batch;
    batch.push_back(tensor_image);

    auto result = model.forward(batch);
    torch::Tensor disp_tensor = result.toTensor();
    cout << "got the disp tensor!!!" << endl;
    disp_tensor = disp_tensor.permute({0, 2, 3, 1});

    disp_tensor = disp_tensor.to(at::kCPU);

    cv::Mat disp = cv::Mat(height, width, CV_32FC1, disp_tensor.data_ptr());
    
    //linear transform
    double minVal;
    double maxVal;
    cv::minMaxLoc(disp, &minVal, &maxVal);
    disp /= maxVal;
    cv::resize(disp, disp, cv::Size(image.cols, image.rows));
    disp*= 255;

    disp.convertTo(disp, CV_8UC1);
    cv::cvtColor(disp, disp, cv::COLOR_GRAY2BGR);
    return disp;
}

void MonoDepth::disp2Depth(cv::Mat &dispMap, cv::Mat &depthMap)
{
    int type = dispMap.type();
    if (type == CV_8UC1){
        uchar* dispData = (uchar*)dispMap.data;
        ushort* depthData = (ushort*)depthMap.data;
        for (int i = 0; i < dispMap.rows; i++)
        {
            for (int j = 0; j < dispMap.cols; j++)
            {
                int id = i*dispMap.cols + j;
                if (!dispData[id])  continue;  //防止0除
                depthData[id] = ushort( (float)1.f / ((float)dispData[id]) );
            }
        }
        cout<<depthMap<<"done"<<endl;
    }
    else if(type == CV_32FC1){
        float* dispData = (float*)dispMap.data;
        ushort* depthData = (ushort*)depthMap.data;
        for(int i = 0; i < dispMap.rows; i++){
            for(int j = 0; j < dispMap.cols; j++){
                int id = (int) i*dispMap.cols + j;
                if (!dispData[id])  continue;  //防止0除
                depthData[id] = ushort( (float)1.f / ((float)dispData[id]));
            }
        }

    }
    else
    {
        cout << "please confirm dispImg's type!" << endl;
        cv::waitKey(0);
    }
}

void insertDepth32f(cv::Mat& depth)
{
    const int width = depth.cols;
    const int height = depth.rows;
    float* data = (float*)depth.data;
    cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
    double* integral = (double*)integralMap.data;
    int* ptsIntegral = (int*)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            if (data[id2] > 1e-3)
            {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }
    // 积分区间
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 1; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd = 2;
    while (dWnd > 1)
    {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i)
        {
            int id1 = i * width;
            for (int j = 0; j < width; ++j)
            {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = max(0, left);
                right = min(right, width - 1);
                top = max(0, top);
                bot = min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0)
                {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
        int s = wnd / 2 * 2 + 1;
        if (s > 201)
        {
            s = 201;
        }
        cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
    }
}

}
// namespace monodepth
