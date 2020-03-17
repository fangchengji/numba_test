#include <normalize.h>

#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
#include <ctime>

int main( int argc, char** argv ) {
    if( argc != 2) {
        std::cout <<" Usage: display_image ImageToLoadAndDisplay" << std::endl;
        return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data ) {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    std::cout << "load image successed!!\n";

    cv::Size sz = image.size();
    std::cout << "image width:" << sz.width << ", height " << sz.height << ", channel " << image.channels()<< " \n";

    // convert to float32
    cv::Mat img;
    image.convertTo(img, CV_32F);
    std::cout << float(img.at<cv::Vec3f>(1, 0)[0]) << " " \
        << float(img.at<cv::Vec3f>(1, 0)[1]) << " " \
        << float(img.at<cv::Vec3f>(1, 0)[2]) << std::endl;
    std::cout << *(img.ptr<float>() + 2400) << " " \
        << *(img.ptr<float>() + 2400 +  1) << " " \
        << *(img.ptr<float>() + 2400 +  2) << std::endl;
    
    float mean[3] = {103.530, 116.280, 123.675};
    float std[3] = {1.0, 1.0, 1.0};
    
    // Allocate device memory.
    float* d_input = nullptr;
    float* d_output = nullptr;
    const int input_bytes = image.channels() * image.rows * image.cols * sizeof(float);
    cudaMalloc((void**)&d_input, input_bytes);
    cudaMalloc((void**)&d_output, input_bytes);

    float* d_mean = nullptr;
    float* d_std = nullptr;
    const int mean_bytes = image.channels()*sizeof(float);
    cudaMalloc((void**)&d_mean, mean_bytes);
    cudaMalloc((void**)&d_std, mean_bytes); 
    cudaMemcpy(d_mean, mean, mean_bytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_std, std, mean_bytes, cudaMemcpyHostToDevice);

    float* h_out = new float[input_bytes/sizeof(float)];

    clock_t t;
    t = clock();
    // Copy data from input image to device memory.
    cudaMemcpy(d_input, img.ptr<float>(), input_bytes, cudaMemcpyHostToDevice);
    // t = clock() - t;
    // std::cout << "data copy time " << float(t)/CLOCKS_PER_SEC * 1000 << "ms\n";

    shopee::ds::GpuTimer timer;

    timer.start();
    // img.cols = width, img.rows = height
    normalize(d_input, d_output, img.cols, img.rows, img.channels(), d_mean, d_std);
    timer.stop();
    
    // t = clock();
    cudaMemcpy(h_out, d_output, input_bytes, cudaMemcpyDeviceToHost);

    t = clock() - t;
    std::cout << "total time " << float(t)/CLOCKS_PER_SEC * 1000.0 << "ms\n";
    std::cout << "gpu normalize time " << timer.elapsed() << "ms\n";

    cv::Mat final_img(cv::Size(image.cols, image.rows), CV_32FC3, h_out);
    cv::imwrite("final.jpg", final_img);

    std::cout << float(final_img.at<cv::Vec3f>(1, 0)[0]) << " " \
        << float(final_img.at<cv::Vec3f>(1, 0)[1]) << " " \
        << float(final_img.at<cv::Vec3f>(1, 0)[2]) << std::endl;

    delete[] h_out;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_std);

    return 0;
}
