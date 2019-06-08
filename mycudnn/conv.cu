#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

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


cv::Mat load_image(
        const char *image_path, cudnnTensorFormat_t format, int H, int W) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    std::cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
              << image.channels() << std::endl;
    if (format == CUDNN_TENSOR_NHWC) {
        return image;
    } else if (format == CUDNN_TENSOR_NCHW) {
        cv::Mat inputBlob = cv::dnn::blobFromImage(
                /*image*/image,
                /*scale_factor*/1.0f,
                ///*size*/cv::Size(image.rows, image.cols),
                /*size*/cv::Size(H, W),
                /*mean*/cv::Scalar(0, 0, 0),
                /*swapRB*/false,
                /*crop*/true);
        return inputBlob;
    } else {
        std::cerr << "Unknown format: " << format << std::endl;
        exit(1);
    }

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


    const int max_kernel_height = 13;
    const int max_kernel_width = 13;


    int in_batch_size = 1;
    int H = 224;
    int W = 224;
    int in_channels = 3;
    int out_channels = 3;
    int kernel_height = 3;
    int kernel_width = 3;
    int pad_height = 1;
    int pad_width = 1;
    int vertical_stride = 1;
    int horizontal_stride = 1;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC;

    bool tests = true;
    if (tests == true) {
        in_batch_size = 128;
        H = 224;
        W = 224;
        in_channels = 3;
        out_channels = 64;
        kernel_height = 7;
        kernel_width = 7;
        pad_height = int(kernel_height / 2);
        pad_width = int(kernel_width / 2);
        vertical_stride = 1;
        horizontal_stride = 1;
        format = CUDNN_TENSOR_NCHW;
    }

    std::cerr
    << "in batch size: " << in_batch_size
    << " H: " << H
    << " W: " << W
    << " in_channels: " << in_channels
    << " out_channels: " << out_channels
    << " kernel_height: " << kernel_height
    << " kernel_width: " << kernel_width
    << " vertical stride: " << vertical_stride
    << "horizontal stride: " << horizontal_stride
    << " format: " << format
    << std::endl;

    /*
    0 CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        This algorithm expresses the convolution as a matrix product without
        actually explicitly form the matrix that holds the input tensor data.
    1 CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        This algorithm expresses the convolution as a matrix product without
        actually explicitly form the matrix that holds the input tensor data,
        but still needs some memory workspace to precompute some indices in
        order to facilitate the implicit construction of the matrix that holds
        the input tensor data.
    2 CUDNN_CONVOLUTION_FWD_ALGO_GEMM
        This algorithm expresses the convolution as an explicit matrix product.
        A significant memory workspace is needed to store the matrix that holds
        the input tensor data.
    3 CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
        This algorithm expresses the convolution as a direct convolution (e.g
        without implicitly or explicitly doing a matrix multiplication).
    4 CUDNN_CONVOLUTION_FWD_ALGO_FFT
        This algorithm uses the Fast-Fourier Transform approach to compute the
        convolution. A significant memory workspace is needed to store
        intermediate results.
    5 CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
        This algorithm uses the Fast-Fourier Transform approach but splits the
        inputs into tiles. A significant memory workspace is needed to store
        intermediate results but less than CUDNN_CONVOLUTION_FWD_ALGO_FFT for
        large size images.
    6 CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
        This algorithm uses the Winograd Transform approach to compute the
        convolution. A reasonably sized workspace is needed to store
        intermediate results.
    7 CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
        This algorithm uses the Winograd Transform approach to compute the
        convolution. Significant workspace may be needed to store intermediate
        results.
     */
    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    std::cout << "format: " << format << std::endl;
    cv::Mat image = load_image(argv[1], format, H, W);

    std::cerr << "Input image after loading: " << image.rows << " x "
              << image.cols << " x " << image.channels() << std::endl;

    cudaSetDevice(gpu_id);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
            /*format=*/format,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/in_batch_size,
            /*channels=*/in_channels,
            /*image_height=*/H,
            /*image_width=*/W));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/out_channels,
            /*in_channels=*/in_channels,
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

    std::cerr << "Output Image: " << batch_size << " x " << height
              << " x "
              << width << " x " << channels << std::endl;

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
            /*format=*/format,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/batch_size,
            /*channels=*/channels,
            /*image_height=*/height,
            /*image_width=*/width));

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

    if (convolution_algorithm < 0) {
        checkCUDNN(
                cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                        /*memoryLimitInBytes=*/0,
                                                    &convolution_algorithm));
    }

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

    int image_bytes = in_channels * height * width * sizeof(float);

    float *d_input{nullptr};
    cudaMalloc(&d_input, image_bytes * batch_size);
    // Copy the same image batch_size number.
    for (int i = 0; i < batch_size; ++i)
        cudaMemcpy(d_input + i * image_bytes, image.ptr<float>(0), image_bytes,
                   cudaMemcpyHostToDevice);

    int output_size = batch_size * height * width * channels * sizeof(float);
    float *d_output{nullptr};
    cudaMalloc(&d_output, output_size);
    cudaMemset(d_output, 0, output_size);

    // clang-format off
    const float kernel_template[max_kernel_height][max_kernel_width] = {
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.5,  1.5,  2.0, 3.1, 5.3,  6.3},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, -8.9, 10.5, 4.3, 5.3, -1.3, -2.3},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.6,  9.5,  5.3, 1.3, 2.3,  -0.31},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.1,  -1.5, 2.1, 3.1, 5.1,  6.1},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, -8.9, 2.5,  4.1, 5.1, -1.1, -2.1},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.2,  3.5,  5.1, 1.1, 2.1,  -1.1},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.3,  1.5,  2.1, 3.1, 5.2,  6.1},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.5,  1.5,  2.0, 3.1, 5.3,  6.3},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.5,  1.5,  2.0, 3.1, 5.3,  6.3},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.5,  1.5,  2.0, 3.1, 5.3,  6.3},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.5,  1.5,  2.0, 3.1, 5.3,  6.3},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.5,  1.5,  2.0, 3.1, 5.3,  6.3},
            {2.3, 4.1, 1.1, 8.1, -1.2, 0.9, 1.7, 1.5,  1.5,  2.0, 3.1, 5.3,  6.3},
    };
    // clang-format on

    float h_kernel[out_channels][in_channels][kernel_height][kernel_width];
    for (int kernel = 0; kernel < out_channels; ++kernel) {
        for (int channel = 0; channel < in_channels; ++channel) {
            for (int row = 0; row < kernel_width; ++row) {
                for (int column = 0; column < kernel_height; ++column) {
                    h_kernel[kernel][channel][row][column] = kernel_template[row][column];
                }
            }
        }
    }

    float *d_kernel{nullptr}; // d stands for device memory
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start));
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
    HANDLE_ERROR(cudaEventRecord(stop));

    HANDLE_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "cuda elapsed time (ms): " << milliseconds << std::endl;

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

    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaEventDestroy(start));

    delete[] h_output;
    HANDLE_ERROR(cudaFree(d_kernel));
    HANDLE_ERROR(cudaFree(d_input));
    HANDLE_ERROR(cudaFree(d_output));
//  cudaFree(d_perfResults);
    HANDLE_ERROR(cudaFree(d_workspace));

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}