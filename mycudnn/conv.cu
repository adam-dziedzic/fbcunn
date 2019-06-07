#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }                                                          \


static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))


cv::Mat load_image(const char *image_path) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    std::cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
              << image.channels() << std::endl;
    return image;
}

void save_image(const char *output_filename,
                float *buffer,
                int height,
                int width) {
    cv::Mat output_image(height, width, CV_32FC3, buffer);
    // Make negative values zero.
    cv::threshold(output_image,
                  output_image,
            /*threshold=*/0,
            /*maxval=*/0,
                  cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC3);
    cv::imwrite(output_filename, output_image);
    std::cerr << "Wrote output to " << output_filename << std::endl;
}

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        std::cerr << "usage: conv <image> [gpu=0] [sigmoid=0]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
    std::cerr << "GPU: " << gpu_id << std::endl;

    bool with_sigmoid = (argc > 3) ? std::atoi(argv[3]) : 0;
    std::cerr << "With sigmoid: " << std::boolalpha << with_sigmoid
              << std::endl;


    int in_batch_size = 1;
    int out_channels = 3;
    int kernel_height = 3;
    int kernel_width = 3;
    int pad_height = 1;
    int pad_width = 1;
    int vertical_stride = 1;
    int horizontal_stride = 1;

    bool tests = false;
    if (tests == true) {
        in_batch_size = 32;
        out_channels = 64;
        kernel_height = 7;
        kernel_width = 7;
        pad_height = 3;
        pad_width = 3;
        vertical_stride = 2;
        horizontal_stride = 2;
    }

    cv::Mat image = load_image(argv[1]);

    cudaSetDevice(gpu_id);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/in_batch_size,
            /*channels=*/3,
            /*image_height=*/image.rows,
            /*image_width=*/image.cols));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/out_channels,
            /*in_channels=*/3,
            /*kernel_height=*/kernel_height,
            /*kernel_width=*/kernel_width));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
            /*pad_height=*/pad_height,
            /*pad_width=*/pad_width,
            /*vertical_stride=*/vertical_stride,
            /*horizontal_stride=*/horizontal_stride,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/CUDNN_DATA_FLOAT));

    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     &batch_size,
                                                     &channels,
                                                     &height,
                                                     &width));

    std::cerr << "Output Image: " << height << " x " << width << " x "
              << channels
              << std::endl;

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/3,
            /*image_height=*/image.rows,
            /*image_width=*/image.cols));

    const int requestedAlgoCount = 8;
    int returnedAlgoCount = 8;
//  cudnnConvolutionFwdAlgoPerf_t perfResults;
    cudnnConvolutionFwdAlgoPerf_t perfResults[requestedAlgoCount];
//  HANDLE_ERROR(cudaMalloc((void**)&perfResults,
//          sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount));

    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
            /*cudnnHandle_t*/cudnn,
            /*cudnnTensorDescriptor_t*/input_descriptor,
            /*cudnnFilterDescriptor_t*/kernel_descriptor,
            /*cudnnConvolutionDescriptor_t*/convolution_descriptor,
            /*cudnnTensorDescriptor_t*/output_descriptor,
            /*int*/requestedAlgoCount,
            /*int**/&returnedAlgoCount,
            /*cudnnConvolutionFwdAlgoPerf_t*/perfResults));

    std::cout << "returnedAlgoCount: " << returnedAlgoCount << std::endl;

    for (int i = 0; i < returnedAlgoCount; ++i) {
        std::cout << "perfResults: " << i << " algo " << perfResults[i].algo
                  << " time: " << perfResults[i].time
                  << " memory: " << perfResults[i].memory
                  << " status: " << perfResults[i].status
                  << " determinism: " << perfResults[i].determinism
                  << " math_type: " << perfResults[i].mathType
                  << std::endl;
    }

//  for (int i = 0; i < returnedAlgoCount; ++i) {
//      std::cout << "# convolution algorithm: " << d_perfResults[i].algo
//                << " time: " << d_perfResults[i].time
//                << " memory: " << d_perfResults[i].memory
//                << std::endl;
//  }

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                input_descriptor,
                                                kernel_descriptor,
                                                convolution_descriptor,
                                                output_descriptor,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm));

    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t
    std::cout << "# convolution algorithm: " << convolution_algorithm
              << std::endl;

    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       input_descriptor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       convolution_algorithm,
                                                       &workspace_bytes));
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
              << std::endl;
    assert(workspace_bytes > 0);

    void *d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    int image_bytes = batch_size * channels * height * width * sizeof(float);

    float *d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, image.ptr<float>(0), image_bytes,
               cudaMemcpyHostToDevice);

    float *d_output{nullptr};
    cudaMalloc(&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    // clang-format off
    const float kernel_template[3][3] = {
            {1, 1,  1},
            {1, -8, 1},
            {1, 1,  1}
    };
    // clang-format on

    float h_kernel[3][3][3][3];
    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    h_kernel[kernel][channel][row][column] = kernel_template[row][column];
                }
            }
        }
    }

    float *d_kernel{nullptr};
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_descriptor,
                                       d_input,
                                       kernel_descriptor,
                                       d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       d_output));

    if (with_sigmoid) {
        cudnnActivationDescriptor_t activation_descriptor;
        checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
        checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                                CUDNN_ACTIVATION_SIGMOID,
                                                CUDNN_PROPAGATE_NAN,
                /*relu_coef=*/0));
        checkCUDNN(cudnnActivationForward(cudnn,
                                          activation_descriptor,
                                          &alpha,
                                          output_descriptor,
                                          d_output,
                                          &beta,
                                          output_descriptor,
                                          d_output));
        cudnnDestroyActivationDescriptor(activation_descriptor);
    }

    float *h_output = new float[image_bytes];
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

    save_image("cudnn-out.png", h_output, height, width);

    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
//  cudaFree(d_perfResults);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}