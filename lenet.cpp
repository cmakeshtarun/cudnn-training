/*
 * This code is released into the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <mpi.h>

#include "readubyte.h"

///////////////////////////////////////////////////////////////////////////////////////////
// Definitions and helper utilities

// Block width for CUDA kernels
#define BW 128

#ifdef U
SE_GFLAGS
    #include <gflags/gflags.h>

    #ifndef _WIN32
        #define gflags google
    #endif
#else
    // Constant versions of gflags
    #define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
    #define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
    #define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
    #define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
    #define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))
#endif


/**
 * Saves a PGM grayscale image out of unsigned 8-bit data
 */
void SavePGMFile(const unsigned char *data, size_t width, size_t height, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (fp)
    {
        fprintf(fp, "P5\n%lu %lu\n255\n", width, height);
        fwrite(data, sizeof(unsigned char), width * height, fp);
        fclose(fp);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code 
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


///////////////////////////////////////////////////////////////////////////////////////////
// Command-line flags

// Application parameters
DEFINE_int32(gpu, 0, "The GPU ID to use");
DEFINE_int32(iterations, 1000, "Number of iterations for training");
DEFINE_int32(random_seed, -1, "Override random seed (default uses std::random_device)");
DEFINE_int32(classify, -1, "Number of images to classify to compute error rate (default uses entire test set)");

// Batch parameters
DEFINE_uint64(batch_size, 64, "Batch size for training");

// Filenames
DEFINE_bool(pretrained, false, "Use the pretrained CUDNN model as input");
DEFINE_bool(save_data, false, "Save pretrained weights to file");
DEFINE_string(train_images, "train-images-idx3-ubyte", "Training images filename");
DEFINE_string(train_labels, "train-labels-idx1-ubyte", "Training labels filename");
DEFINE_string(test_images, "t10k-images-idx3-ubyte", "Test images filename");
DEFINE_string(test_labels, "t10k-labels-idx1-ubyte", "Test labels filename");

// Solver parameters
DEFINE_double(learning_rate, 0.01, "Base learning rate");
DEFINE_double(lr_gamma, 0.0001, "Learning rate policy gamma");
DEFINE_double(lr_power, 0.75, "Learning rate policy power");

void launch_FillOnes(int bs, int bw, float *vec);

void launch_SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff, int bw);

// FLAGS for MPI communication
// enum Flags{ COMM_XDATA, COMM_XLABEL, COMM_HEIGHT, COMM_WIDTH, COMM_TRAIN_SIZE, COMM_TRAIN_IMAGES_SIZE, 
//		COMM_GCONV1, COMM_GCONV1BIAS, COMM_GCONV2, COMM_GCONV2BIAS, COMM_GFC1NEURON, COMM_GFC1BIAS, COMM_GFC2NEURON, COMM_GFC2BIAS,
//		COMM_GDCONV1, COMM_GDCONV1BIAS, COMM_GDCONV2, COMM_GDCONV2BIAS, COMM_GDFC1NEURON, COMM_GDFC1BIAS, COMM_GDFC2NEURON, COMM_GDFC2BIAS
//	};
#define COMM_XDATA 		0
#define COMM_XLABEL		1
#define COMM_HEIGHT		2
#define COMM_WIDTH		3
#define COMM_TRAIN_SIZE		4
#define COMM_TRAIN_IMAGES_SIZE	5
#define COMM_GCONV1		6
#define COMM_GCONV1BIAS		7
#define COMM_GCONV2		8
#define COMM_GCONV2BIAS		9
#define COMM_GFC1NEURON		10
#define COMM_GFC1BIAS		11
#define COMM_GFC2NEURON		12
#define COMM_GFC2BIAS		13
#define COMM_GDCONV1		14
#define COMM_GDCONV1BIAS	15
#define COMM_GDCONV2		16
#define COMM_GDCONV2BIAS	17
#define COMM_GDFC1NEURON	18
#define COMM_GDFC1BIAS		19
#define COMM_GDFC2NEURON	20
#define COMM_GDFC2BIAS		21

///////////////////////////////////////////////////////////////////////////////////////////
// Layer representations

/**
 * Represents a convolutional layer with bias.
 */
struct ConvBiasLayer
{
    int in_channels, out_channels, kernel_size;
    int in_width, in_height, out_width, out_height;

    std::vector<float> pconv, pbias;
    
    ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_, 
                  int in_w_, int in_h_) : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_), 
                  pbias(out_channels_)
    {
        in_channels = in_channels_;
        out_channels = out_channels_;
        kernel_size = kernel_size_;
        in_width = in_w_;
        in_height = in_h_;
        out_width = in_w_ - kernel_size_ + 1;
        out_height = in_h_ - kernel_size_ + 1;
    }

    bool FromFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";
        
        // Read weights file
        FILE *fp = fopen(ssf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            return false;
        }
        fread(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&pbias[0], sizeof(float), out_channels, fp);
        fclose(fp);
        return true;
    }

    void ToFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Write weights file
        FILE *fp = fopen(ssf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            exit(2);
        }
        fwrite(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&pbias[0], sizeof(float), out_channels, fp);
        fclose(fp);
    }
};

/**
 * Represents a max-pooling layer.
 */
struct MaxPoolLayer
{
    int size, stride;
    MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

/**
 * Represents a fully-connected neural network layer with bias.
 */
struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
        pneurons(inputs_ * outputs_), pbias(outputs_) {}

    bool FromFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Read weights file
        FILE *fp = fopen(ssf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            return false;
        }
        fread(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
        return true;
    }

    void ToFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Write weights file
        FILE *fp = fopen(ssf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            exit(2);
        }
        fwrite(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////
// CUDNN/CUBLAS training context

struct TrainingContext
{
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor, 
                             conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
    cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
    cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
    cudnnConvolutionFwdAlgo_t conv1algo, conv2algo;
    cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
    cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnActivationDescriptor_t fc1Activation;

    int m_gpuid;
    int m_batchSize;
    size_t m_workspaceSize;

    FullyConnectedLayer& ref_fc1, &ref_fc2;

    // Disable copying
    TrainingContext& operator=(const TrainingContext&) = delete;
    TrainingContext(const TrainingContext&) = delete;

    TrainingContext(int gpuid, int batch_size,
                    ConvBiasLayer& conv1, MaxPoolLayer& pool1, ConvBiasLayer& conv2, MaxPoolLayer& pool2,
                    FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuid)
    {
        m_batchSize = batch_size;

        // Create CUBLAS and CUDNN handles
        checkCudaErrors(cudaSetDevice(gpuid));
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCUDNN(cudnnCreate(&cudnnHandle));

        // Create tensor descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc2Tensor));

        checkCUDNN(cudnnCreateActivationDescriptor(&fc1Activation));

        checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&conv2filterDesc));

        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2Desc));

        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));            

        
        // Set tensor descriptor sizes
        checkCUDNN(cudnnSetTensor4dDescriptor(conv1BiasTensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              1, conv1.out_channels,
                                              1, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(conv2BiasTensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              1, conv2.out_channels,
                                              1, 1));
            
        checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                                               CUDNN_POOLING_MAX,
                                               CUDNN_PROPAGATE_NAN,
                                               pool1.size, pool1.size,
                                               0, 0,
                                               pool1.stride, pool1.stride));
        checkCUDNN(cudnnSetTensor4dDescriptor(pool2Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, conv2.out_channels,
                                              conv2.out_height / pool2.stride,
                                              conv2.out_width / pool2.stride));

        checkCUDNN(cudnnSetTensor4dDescriptor(fc1Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, fc1.outputs, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(fc2Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, fc2.outputs, 1, 1));

        checkCUDNN(cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN, 0.0));


        // Set convolution tensor sizes and compute workspace size
        size_t workspace = 0;
        workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

        workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));

        // The workspace is allocated later (if necessary)
        m_workspaceSize = workspace;
    }

    ~TrainingContext()
    {
        checkCudaErrors(cudaSetDevice(m_gpuid));

        checkCudaErrors(cublasDestroy(cublasHandle));
        checkCUDNN(cudnnDestroy(cudnnHandle));
        checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv1BiasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(pool1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv2Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv2BiasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(pool2Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(fc1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(fc2Tensor));
        checkCUDNN(cudnnDestroyActivationDescriptor(fc1Activation));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv2filterDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2Desc));
        checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    }

    size_t SetFwdConvolutionTensors(ConvBiasLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                    cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc, 
                                    cudnnConvolutionFwdAlgo_t& algo)
    {
        size_t sizeInBytes = 0;

        int n = m_batchSize;
        int c = conv.in_channels;
        int h = conv.in_height;
        int w = conv.in_width;

        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              n, c,
                                              h, w));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                              CUDNN_DATA_FLOAT,
                                              CUDNN_TENSOR_NCHW,
                                              conv.out_channels,
                                              conv.in_channels, 
                                              conv.kernel_size,
                                              conv.kernel_size));

#if CUDNN_MAJOR > 5
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,
                                                   1, 1,
                                                   1, 1,
                                                   CUDNN_CROSS_CORRELATION,
                                                   CUDNN_DATA_FLOAT));
#else
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,
                                                   1, 1,
                                                   1, 1,
                                                   CUDNN_CROSS_CORRELATION));
#endif

        // Find dimension of convolution output
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                         srcTensorDesc,
                                                         filterDesc,
                                                         &n, &c, &h, &w));

        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              n, c,
                                              h, w));
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                       srcTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                       0,
                                                       &algo));
        
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                           srcTensorDesc,
                                                           filterDesc,
                                                           convDesc,
                                                           dstTensorDesc,
                                                           algo,
                                                           &sizeInBytes));

        return sizeInBytes;
    }

    void ForwardPropagation(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                            float *fc2, float *result,
                            float *pconv1, float *pconv1bias, 
                            float *pconv2, float *pconv2bias, 
                            float *pfc1, float *pfc1bias,
                            float *pfc2, float *pfc2bias, void *workspace, float *onevec)
    {        
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Conv1 layer
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
                                           data, conv1filterDesc, pconv1, conv1Desc, 
                                           conv1algo, workspace, m_workspaceSize, &beta,
                                           conv1Tensor, conv1));
        checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor,
                                  pconv1bias, &alpha, conv1Tensor, conv1));

        // Pool1 layer
        checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,
                                       conv1, &beta, pool1Tensor, pool1));

        // Conv2 layer
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,
                                           pool1, conv2filterDesc, pconv2, conv2Desc, 
                                           conv2algo, workspace, m_workspaceSize, &beta,
                                           conv2Tensor, conv2));
        checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,
                                  pconv2bias, &alpha, conv2Tensor, conv2));

        // Pool2 layer
        checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,
                                       conv2, &beta, pool2Tensor, pool2));

        // FC1 layer
        // Forward propagate neurons using weights (fc1 = pfc1'*pool2)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_fc1.outputs, m_batchSize, ref_fc1.inputs,
                                    &alpha,
                                    pfc1, ref_fc1.inputs,
                                    pool2, ref_fc1.inputs,
                                    &beta,
                                    fc1, ref_fc1.outputs));
        // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_fc1.outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc1bias, ref_fc1.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc1, ref_fc1.outputs));

        // ReLU activation
        checkCUDNN(cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,
                                          fc1Tensor, fc1, &beta, fc1Tensor, fc1relu));

        // FC2 layer
        // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_fc2.outputs, m_batchSize, ref_fc2.inputs,
                                    &alpha,
                                    pfc2, ref_fc2.inputs,
                                    fc1relu, ref_fc2.inputs,
                                    &beta,
                                    fc2, ref_fc2.outputs));
        // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_fc2.outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc2bias, ref_fc2.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc2, ref_fc2.outputs));

        // Softmax loss
        checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                       &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
    }

    size_t SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                    cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc, 
                                    cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo)
    {
        size_t sizeInBytes = 0, tmpsize = 0;

        // If backprop filter algorithm was requested
        if (falgo)
        {
            checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo));

            checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc, 
                *falgo, &tmpsize));

            sizeInBytes = std::max(sizeInBytes, tmpsize);
        }

        // If backprop data algorithm was requested
        if (dalgo)
        {
            checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo));

            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc, 
                *dalgo, &tmpsize));

            sizeInBytes = std::max(sizeInBytes, tmpsize);
        }
        
        return sizeInBytes;
    }

    void Backpropagation(ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
                         float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                         float *fc2, float *fc2smax, float *dloss_data,
                         float *pconv1, float *pconv1bias,
                         float *pconv2, float *pconv2bias,
                         float *pfc1, float *pfc1bias,
                         float *pfc2, float *pfc2bias,
                         float *gconv1, float *gconv1bias, float *dpool1,
                         float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
                         float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
                         float *gfc2, float *gfc2bias, float *dfc2,
                         void *workspace, float *onevec)
    {    
        float alpha = 1.0f, beta = 0.0f;

        float scalVal = 1.0f / static_cast<float>(m_batchSize);

        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Initialization (using the training error function)
        checkCudaErrors(cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float) * m_batchSize * ref_fc2.outputs, cudaMemcpyDeviceToDevice));
        
        // Softmax layer
        launch_SoftmaxLossBackprop(labels, ref_fc2.outputs, m_batchSize, dloss_data, BW);

        // Accounting for batch size in SGD
        checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));

        // FC2 layer
        // Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, m_batchSize,
                                    &alpha, fc1relu, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs));
        // Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.outputs, m_batchSize,
                                    &alpha, dloss_data, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, m_batchSize, ref_fc2.outputs,
                                    &alpha, pfc2, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs));
        
        // ReLU activation
        checkCUDNN(cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,
                                           fc1Tensor, fc1relu, fc1Tensor, dfc2,
                                           fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu));

        // FC1 layer
        // Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, m_batchSize,
                                    &alpha, pool2, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs));
        // Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.outputs, m_batchSize,
                                    &alpha, dfc1relu, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, m_batchSize, ref_fc1.outputs,
                                    &alpha, pfc1, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs));

        // Pool2 layer
        checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, 
                                        pool2Tensor, pool2, pool2Tensor, dfc1,
                                        conv2Tensor, conv2, &beta, conv2Tensor, dpool2));
        
        // Conv2 layer
        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,
                                                dpool2, &beta, conv2BiasTensor, gconv2bias));

        
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor,
                                                  pool1, conv2Tensor, dpool2, conv2Desc,
                                                  conv2bwfalgo, workspace, m_workspaceSize,
                                                  &beta, conv2filterDesc, gconv2));
    
        checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc,
                                                pconv2, conv2Tensor, dpool2, conv2Desc, 
                                                conv2bwdalgo, workspace, m_workspaceSize,
                                                &beta, pool1Tensor, dconv2));
        
        // Pool1 layer
        checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, 
                                        pool1Tensor, pool1, pool1Tensor, dconv2,
                                        conv1Tensor, conv1, &beta, conv1Tensor, dpool1));
        
        // Conv1 layer
        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
                                                dpool1, &beta, conv1BiasTensor, gconv1bias));
        
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
                                                  data, conv1Tensor, dpool1, conv1Desc,
                                                  conv1bwfalgo, workspace, m_workspaceSize,
                                                  &beta, conv1filterDesc, gconv1));

        // No need for convBackwardData because there are no more layers below
    }

    void UpdateLocalWeights(float learning_rate, float rho,
                       ConvBiasLayer& conv1, ConvBiasLayer& conv2,
                       float *gpconv1, float *gpconv1bias,
                       float *gpconv2, float *gpconv2bias,
                       float *gpfc1, float *gpfc1bias,
                       float *gpfc2, float *gpfc2bias,
                       float *gdpconv1, float *gdpconv1bias,
                       float *gdpconv2, float *gdpconv2bias,
                       float *gdpfc1, float *gdpfc1bias,
                       float *gdpfc2, float *gdpfc2bias,
                       float *pconv1, float *pconv1bias,
                       float *pconv2, float *pconv2bias,
                       float *pfc1, float *pfc1bias,
                       float *pfc2, float *pfc2bias,
                       float *gconv1, float *gconv1bias,
                       float *gconv2, float *gconv2bias,
                       float *gfc1, float *gfc1bias,
                       float *gfc2, float *gfc2bias)
    {    
        float alpha = -learning_rate;
	float rho_alpha = -rho*alpha;
        float minus_rho_alpha = rho_alpha;
        float minus_one = -1;

        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Conv1
        checkCudaErrors(cudaMemset(gdpconv1, 0, sizeof(float) * conv1.pconv.size()));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
				    &rho_alpha, pconv1, 1, gdpconv1, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
				    &minus_rho_alpha, gpconv1, 1, gdpconv1, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
				    &minus_one, gdpconv1, 1, pconv1, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
                                    &alpha, gpconv1, 1, pconv1, 1));

	// Conv1 bias
        checkCudaErrors(cudaMemset(gdpconv1bias, 0, sizeof(float) * conv1.pbias.size()));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
				    &rho_alpha, pconv1bias, 1, gdpconv1bias, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
				    &minus_rho_alpha, gpconv1bias, 1, gdpconv1bias, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
				    &minus_one, gdpconv1bias, 1, pconv1bias, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
                                    &alpha, gconv1bias, 1, pconv1bias, 1));

        // Conv2
        checkCudaErrors(cudaMemset(gdpconv2, 0, sizeof(float) * conv2.pconv.size()));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
				    &rho_alpha, pconv2, 1, gdpconv2, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
				    &minus_rho_alpha, gpconv2, 1, gdpconv2, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
				    &minus_one, gdpconv2, 1, pconv2, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
                                    &alpha, gconv2, 1, pconv2, 1));
        // Conv2 bias
        checkCudaErrors(cudaMemset(gdpconv2bias, 0, sizeof(float) * conv2.pbias.size()));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
				    &rho_alpha, pconv2bias, 1, gdpconv2bias, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
				    &minus_rho_alpha, gpconv2, 1, gdpconv2bias, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
				    &minus_one, gdpconv2bias, 1, pconv2bias, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
                                    &alpha, gconv2bias, 1, pconv2bias, 1));

        // Fully connected 1
        checkCudaErrors(cudaMemset(gfc1, 0, sizeof(float) * ref_fc1.pneurons.size()));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
				    &rho_alpha, pfc1, 1, gdpfc1, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
				    &minus_rho_alpha, gpfc1, 1, gdpfc1, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
				    &minus_one, gdpfc1, 1, pfc1, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
                                    &alpha, gfc1, 1, pfc1, 1));

        // Fully connected 1 bias
        checkCudaErrors(cudaMemset(gdpfc1bias, 0, sizeof(float) * ref_fc1.pbias.size()));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
				    &rho_alpha, pfc1bias, 1, gdpfc1bias, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
				    &minus_rho_alpha, gpfc1bias, 1, gdpfc1bias, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
				    &minus_one, gdpfc1bias, 1, pfc1bias, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
                                    &alpha, gfc1bias, 1, pfc1bias, 1));

        // Fully connected 2
        checkCudaErrors(cudaMemset(gdpfc2, 0, sizeof(float) * ref_fc2.pneurons.size()));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
				    &rho_alpha, pfc2, 1, gdpfc2, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
				    &minus_rho_alpha, gpfc2, 1, gdpfc2, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
				    &minus_one, gdpfc2, 1, pfc2, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
                                    &alpha, gfc2, 1, pfc2, 1));

        // Fully connected 2 bias
        checkCudaErrors(cudaMemset(gdpfc2bias, 0, sizeof(float) * ref_fc2.pbias.size()));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
				    &rho_alpha, pfc2bias, 1, gdpfc2bias, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
				    &minus_rho_alpha, gpfc2bias, 1, gdpfc2bias, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
				    &minus_one, gdpfc2bias, 1, pfc2bias, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
                                    &alpha, gfc2bias, 1, pfc2bias, 1));
    }


    void UpdateGlobalWeights(float learning_rate,
                       ConvBiasLayer& conv1, ConvBiasLayer& conv2,
                       float *pconv1, float *pconv1bias,
                       float *pconv2, float *pconv2bias,
                       float *pfc1, float *pfc1bias,
                       float *pfc2, float *pfc2bias,
                       float *gconv1, float *gconv1bias,
                       float *gconv2, float *gconv2bias,
                       float *gfc1, float *gfc1bias,
                       float *gfc2, float *gfc2bias)
    {
        float alpha = learning_rate;

        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Conv1
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
                                          &alpha, gconv1, 1, pconv1, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
                                          &alpha, gconv1bias, 1, pconv1bias, 1));
        
        // Conv2
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
                                          &alpha, gconv2, 1, pconv2, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
                                          &alpha, gconv2bias, 1, pconv2bias, 1));
        
        // Fully connected
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
         				  &alpha, gfc1, 1, pfc1, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
                                          &alpha, gfc1bias, 1, pfc1bias, 1));
        	
        // Fully connected 2
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
                                          &alpha, gfc2, 1, pfc2, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
                                          &alpha, gfc2bias, 1, pfc2bias, 1));
    }
        
};


///////////////////////////////////////////////////////////////////////////////////////////
// Main function

int main(int argc, char **argv)
{

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm);

        MPI_Comm_rank(local_comm, &local_rank);

        MPI_Comm_free(&local_comm);
    }

    //cudaSetDevice(local_rank);
    //FLAGS_gpu = local_rank;

#ifdef USE_GFLAGS
    gflags::ParseCommandLineFlags(&argc, &argv, true);
#endif

    size_t width, height, channels = 1;
    size_t train_size, test_size, train_images_size;
    float *train_images_float, *train_labels_float;
    std::vector<uint8_t> train_images, train_labels;
    std::vector<uint8_t> test_images, test_labels;

    if(rank == 0){

        // Open input data
        printf("Reading input data\n");
        
        // Read dataset sizes
        train_size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), nullptr, nullptr, width, height);
        test_size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr, nullptr, width, height);
        if (train_size == 0)
            return 1;        

    	train_images.resize(train_size * width * height * channels);
	train_labels.resize(train_size);
    	test_images.resize(test_size * width * height * channels);
	test_labels.resize(test_size);

        // Read data from datasets
        if (ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size)
            return 2;
        if (ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size)
            return 3;
        printf("width = %d, height = %d\n",width,height);
    
        printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
        printf("Batch size: %lld, iterations: %d\n", FLAGS_batch_size, FLAGS_iterations);

	train_images_size = train_images.size();     
    	train_images_float = (float*) malloc(sizeof(float)*train_images.size());
	train_labels_float = (float*) malloc(sizeof(float)*train_size);

	printf("Preparing dataset\n");
        // Normalize training set to be in [0,1]
        for (size_t i = 0; i < train_size * channels * width * height; ++i)
            train_images_float[i] = (float)train_images[i] / 255.0f;
        
        for (size_t i = 0; i < train_size; ++i)
            train_labels_float[i] = (float)train_labels[i];

    }


    //Bcast dataset parameters
    MPI_Bcast(&height, 			1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width,  			1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&train_size,  		1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&train_images_size,  	1, MPI_INT, 0, MPI_COMM_WORLD);

    // Choose GPU
    int num_gpus;
    checkCudaErrors(cudaGetDeviceCount(&num_gpus));
    if (FLAGS_gpu < 0 || FLAGS_gpu >= num_gpus)
    {
        printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n",
               FLAGS_gpu, num_gpus);
        return 4;
    }

    // Create the LeNet network architecture
    ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
    MaxPoolLayer pool1(2, 2);
    ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
    MaxPoolLayer pool2(2, 2);
    FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 
                            500);
    FullyConnectedLayer fc2(fc1.outputs, 10);

    // Initialize CUDNN/CUBLAS training context
    TrainingContext context(FLAGS_gpu, FLAGS_batch_size, conv1, pool1, conv2, pool2, fc1, fc2);
    
    // Determine initial network structure
    bool bRet = true;
    if (FLAGS_pretrained)
    {
      bRet = conv1.FromFile("conv1");
      bRet &= conv2.FromFile("conv2");
      bRet &= fc1.FromFile("ip1");
      bRet &= fc2.FromFile("ip2");
    }
    if (!bRet || !FLAGS_pretrained)
    {
        // Create random network
        std::random_device rd;
        std::mt19937 gen(FLAGS_random_seed < 0 ? rd() : static_cast<unsigned int>(FLAGS_random_seed));

        // Xavier weight filling
        float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
        std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
        float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
        std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
        float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
        std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
        float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
        std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

        // Randomize network
        for (auto&& iter : conv1.pconv)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv1.pbias)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv2.pconv)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : conv2.pbias)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : fc1.pneurons)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc1.pbias)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc2.pneurons)
            iter = static_cast<float>(dfc2(gen));
        for (auto&& iter : fc2.pbias)
            iter = static_cast<float>(dfc2(gen));
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // Create GPU data structures    

    // Forward propagation data
    float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
    //                         Buffer    | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    checkCudaErrors(cudaMalloc(&d_data,    sizeof(float) * context.m_batchSize * channels           * height                            * width));
    checkCudaErrors(cudaMalloc(&d_labels,  sizeof(float) * context.m_batchSize * 1                  * 1                                 * 1));
    checkCudaErrors(cudaMalloc(&d_conv1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    checkCudaErrors(cudaMalloc(&d_pool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    checkCudaErrors(cudaMalloc(&d_conv2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    checkCudaErrors(cudaMalloc(&d_pool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
    checkCudaErrors(cudaMalloc(&d_fc1,     sizeof(float) * context.m_batchSize * fc1.outputs));    
    checkCudaErrors(cudaMalloc(&d_fc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
    checkCudaErrors(cudaMalloc(&d_fc2,     sizeof(float) * context.m_batchSize * fc2.outputs));
    checkCudaErrors(cudaMalloc(&d_fc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));    

    //Local Network parameters
    float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
    float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
    
    checkCudaErrors(cudaMalloc(&d_pconv1,     sizeof(float) * conv1.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_pconv1bias, sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pconv2,     sizeof(float) * conv2.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_pconv2bias, sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pfc1,       sizeof(float) * fc1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pfc1bias,   sizeof(float) * fc1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pfc2,       sizeof(float) * fc2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pfc2bias,   sizeof(float) * fc2.pbias.size()));    
    
    //Global Network parameters
    //Device objects
    float *d_gpconv1, *d_gpconv1bias, *d_gpconv2, *d_gpconv2bias;
    float *d_gpfc1, *d_gpfc1bias, *d_gpfc2, *d_gpfc2bias;
    
    checkCudaErrors(cudaMalloc(&d_gpconv1,     sizeof(float) * conv1.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_gpconv1bias, sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gpconv2,     sizeof(float) * conv2.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_gpconv2bias, sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gpfc1,       sizeof(float) * fc1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gpfc1bias,   sizeof(float) * fc1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gpfc2,       sizeof(float) * fc2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gpfc2bias,   sizeof(float) * fc2.pbias.size()));    
    
    //Host objects
    float* h_gpconv1	 = (float*)malloc(sizeof(float) * conv1.pconv.size());
    float* h_gpconv1bias = (float*)malloc(sizeof(float) * conv1.pbias.size());
    float* h_gpconv2	 = (float*)malloc(sizeof(float) * conv2.pconv.size());
    float* h_gpconv2bias = (float*)malloc(sizeof(float) * conv2.pbias.size());
    float* h_gpfc1	 = (float*)malloc(sizeof(float) * fc1.pneurons.size());
    float* h_gpfc1bias	 = (float*)malloc(sizeof(float) * fc1.pbias.size());
    float* h_gpfc2	 = (float*)malloc(sizeof(float) * fc2.pneurons.size());
    float* h_gpfc2bias	 = (float*)malloc(sizeof(float) * fc2.pbias.size());    

    //Global - Local offset network parameters
    float *d_gdpconv1, *d_gdpconv1bias, *d_gdpconv2, *d_gdpconv2bias;
    float *d_gdpfc1, *d_gdpfc1bias, *d_gdpfc2, *d_gdpfc2bias;
    
    checkCudaErrors(cudaMalloc(&d_gdpconv1,     sizeof(float) * conv1.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_gdpconv1bias, sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gdpconv2,     sizeof(float) * conv2.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_gdpconv2bias, sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gdpfc1,       sizeof(float) * fc1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gdpfc1bias,   sizeof(float) * fc1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gdpfc2,       sizeof(float) * fc2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gdpfc2bias,   sizeof(float) * fc2.pbias.size()));    

    //Host objects
    float* h_gdpconv1		= (float*)malloc(sizeof(float) * conv1.pconv.size());
    float* h_gdpconv1bias	= (float*)malloc(sizeof(float) * conv1.pbias.size());
    float* h_gdpconv2		= (float*)malloc(sizeof(float) * conv2.pconv.size());
    float* h_gdpconv2bias	= (float*)malloc(sizeof(float) * conv2.pbias.size());
    float* h_gdpfc1	 	= (float*)malloc(sizeof(float) * fc1.pneurons.size());
    float* h_gdpfc1bias	 	= (float*)malloc(sizeof(float) * fc1.pbias.size());
    float* h_gdpfc2	 	= (float*)malloc(sizeof(float) * fc2.pneurons.size());
    float* h_gdpfc2bias	 	= (float*)malloc(sizeof(float) * fc2.pbias.size());    

    // Network parameter gradients
    float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
    float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
    
    checkCudaErrors(cudaMalloc(&d_gconv1,     sizeof(float) * conv1.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_gconv1bias, sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gconv2,     sizeof(float) * conv2.pconv.size()));
    checkCudaErrors(cudaMalloc(&d_gconv2bias, sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gfc1,       sizeof(float) * fc1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gfc1bias,   sizeof(float) * fc1.pbias.size()));    
    checkCudaErrors(cudaMalloc(&d_gfc2,       sizeof(float) * fc2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gfc2bias,   sizeof(float) * fc2.pbias.size()));
    
    // Differentials w.r.t. data
    float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
    //                         Buffer     | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    checkCudaErrors(cudaMalloc(&d_dpool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    checkCudaErrors(cudaMalloc(&d_dpool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    checkCudaErrors(cudaMalloc(&d_dconv2,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    checkCudaErrors(cudaMalloc(&d_dfc1,     sizeof(float) * context.m_batchSize * fc1.inputs));
    checkCudaErrors(cudaMalloc(&d_dfc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
    checkCudaErrors(cudaMalloc(&d_dfc2,     sizeof(float) * context.m_batchSize * fc2.inputs));
    checkCudaErrors(cudaMalloc(&d_dfc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));
    checkCudaErrors(cudaMalloc(&d_dlossdata,sizeof(float) * context.m_batchSize * fc2.outputs));
    
    // Temporary buffers and workspaces
    float *d_onevec;
    void *d_cudnn_workspace = nullptr;    
    checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float)* context.m_batchSize));
    if (context.m_workspaceSize > 0)
        checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));    

    /////////////////////////////////////////////////////////////////////////////

    // Copy initial local network to device
    checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv2, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice));
    
    // Copy initial global network to device
    checkCudaErrors(cudaMemcpyAsync(d_gpconv1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_gpconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_gpconv2, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_gpconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_gpfc1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_gpfc1bias, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_gpfc2, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_gpfc2bias, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice));

    // Fill one-vector with ones
    launch_FillOnes(context.m_batchSize, BW, d_onevec);

    // Objects to hold mini-batches
    float*  train_images_mBatch_float = (float*) malloc(sizeof(float)*context.m_batchSize*train_images_size/train_size);
    float*  train_labels_mBatch_float = (float*) malloc(sizeof(float)*context.m_batchSize);
    int num_mBatch = floor(train_size/context.m_batchSize);

    printf("Training...\n");

    // Use SGD to train the network
    checkCudaErrors(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < FLAGS_iterations; ++iter)
    {
	printf("In iteration %d\n",iter);

	for(int i = 1; i <= rank; i++){
	    int rand_mbid = rand() % num_mBatch;
	    // Distribute Training images for mini-batches
	    if(rank == 0){
	        MPI_Send(&train_images_float[rand_mbid * context.m_batchSize * width*height*channels], context.m_batchSize * channels * width * height,
			MPI_FLOAT, i, COMM_XDATA, MPI_COMM_WORLD);
	        MPI_Send(&train_labels_float[rand_mbid * context.m_batchSize], context.m_batchSize, MPI_FLOAT, i, COMM_XLABEL, MPI_COMM_WORLD);
 	    }

	    if(rank == i){
	    	MPI_Recv(train_images_mBatch_float, context.m_batchSize * channels * width * height, MPI_FLOAT, 0, COMM_XDATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(train_labels_mBatch_float, context.m_batchSize, MPI_FLOAT, 0, COMM_XLABEL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}

	printf("Rank:%d Iter:%d Forward and Backward propogation \n",rank, iter);
	//Forward and Backward propogation on all worker GPUs
	if(rank != 0){

            // Prepare current batch on device
            checkCudaErrors(cudaMemcpyAsync(d_data, &train_images_mBatch_float,
                                            sizeof(float) * context.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float,
                                            sizeof(float) * context.m_batchSize, cudaMemcpyHostToDevice));
            
            // Forward propagation
            context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, 
                                       d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                       d_cudnn_workspace, d_onevec);
    
            // Backward propagation
            context.Backpropagation(conv1, pool1, conv2, pool2,
                                    d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
                                    d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                    d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias, 
                                    d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);
        }

	if(rank == 0){
	    //Copy global weights from device
            checkCudaErrors(cudaMemcpy(d_gpconv1,	h_gpconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gpconv1bias, 	h_gpconv1bias, sizeof(float) * conv1.pbias.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gpconv2,	h_gpconv2, sizeof(float) * conv2.pconv.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gpconv2bias, 	h_gpconv2bias, sizeof(float) * conv2.pbias.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gpfc1,		h_gpfc1, sizeof(float) * fc1.pneurons.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gpfc1bias,   	h_gpfc1bias, sizeof(float) * fc1.pbias.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gpfc2,		h_gpfc2, sizeof(float) * fc2.pneurons.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gpfc2bias,   	h_gpfc2bias, sizeof(float) * fc2.pbias.size(), cudaMemcpyDeviceToHost));
	}

	printf("Iter:%d Broadcasting global weghts\n",iter);
	//Broadcasting Global weights to everyone
	MPI_Bcast(h_gpconv1,		conv1.pconv.size(), 	MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_gpconv1bias,	conv1.pbias.size(), 	MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_gpconv2,		conv2.pconv.size(), 	MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_gpconv2bias,	conv2.pbias.size(), 	MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_gpfc1,		fc1.pneurons.size(),	MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_gpfc1bias,		fc1.pbias.size(),	MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_gpfc2,		fc2.pneurons.size(), 	MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_gpfc2bias,		fc2.pbias.size(), 	MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Compute learning rate
        float learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));
        //TODO: find rho
        float rho = 10.0; 
    
	printf("Iter:%d Update local weights \n",iter);
	if(rank != 0){
	    //Copy global weights from device
            checkCudaErrors(cudaMemcpy(d_gpconv1,	h_gpconv1, sizeof(float) * conv1.pconv.size(),		cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_gpconv1bias, 	h_gpconv1bias, sizeof(float) * conv1.pbias.size(),	cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_gpconv2,	h_gpconv2, sizeof(float) * conv2.pconv.size(), 		cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_gpconv2bias, 	h_gpconv2bias, sizeof(float) * conv2.pbias.size(), 	cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_gpfc1,		h_gpfc1, sizeof(float) * fc1.pneurons.size(), 		cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_gpfc1bias,   	h_gpfc1bias, sizeof(float) * fc1.pbias.size(), 		cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_gpfc2,		h_gpfc2, sizeof(float) * fc2.pneurons.size(), 		cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_gpfc2bias,   	h_gpfc2bias, sizeof(float) * fc2.pbias.size(), 		cudaMemcpyHostToDevice));	

            // Update weights
            context.UpdateLocalWeights(learningRate, rho, conv1, conv2,
                                  d_gpconv1, d_gpconv1bias, d_gpconv2, d_gpconv2bias, d_gpfc1, d_gpfc1bias, d_gpfc2, d_gpfc2bias,
                                  d_gdpconv1, d_gdpconv1bias, d_gdpconv2, d_gdpconv2bias, d_gdpfc1, d_gdpfc1bias, d_gdpfc2, d_gdpfc2bias,
                                  d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                  d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);

	    //Copy rho(L-G) from device
            checkCudaErrors(cudaMemcpy(d_gdpconv1,	h_gdpconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gdpconv1bias, 	h_gdpconv1bias, sizeof(float) * conv1.pbias.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gdpconv2,	h_gdpconv2, sizeof(float) * conv2.pconv.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gdpconv2bias, 	h_gdpconv2bias, sizeof(float) * conv2.pbias.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gdpfc1,	h_gdpfc1, sizeof(float) * fc1.pneurons.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gdpfc1bias,   	h_gdpfc1bias, sizeof(float) * fc1.pbias.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gdpfc2,	h_gdpfc2, sizeof(float) * fc2.pneurons.size(), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(d_gdpfc2bias,   	h_gdpfc2bias, sizeof(float) * fc2.pbias.size(), cudaMemcpyDeviceToHost));
	}

	for(int i = 1; i <= rank; i++){
	    if(rank == i){
		//Send rho(L-G) to root from every processor
	    	MPI_Send(h_gdpconv1,	conv1.pconv.size(), 	MPI_FLOAT, 0, COMM_GDCONV1,	MPI_COMM_WORLD);
	    	MPI_Send(h_gdpconv1bias,conv1.pbias.size(), 	MPI_FLOAT, 0, COMM_GDCONV1BIAS,	MPI_COMM_WORLD);
	    	MPI_Send(h_gdpconv2,	conv2.pconv.size(), 	MPI_FLOAT, 0, COMM_GDCONV2,	MPI_COMM_WORLD);
	    	MPI_Send(h_gdpconv2bias,conv2.pbias.size(), 	MPI_FLOAT, 0, COMM_GDCONV2BIAS,	MPI_COMM_WORLD);
	    	MPI_Send(h_gdpfc1,	fc1.pneurons.size(),	MPI_FLOAT, 0, COMM_GDFC1NEURON, MPI_COMM_WORLD);
	    	MPI_Send(h_gdpfc1bias,	fc1.pbias.size(),	MPI_FLOAT, 0, COMM_GDFC1BIAS, 	MPI_COMM_WORLD);
	    	MPI_Send(h_gdpfc2,	fc2.pneurons.size(), 	MPI_FLOAT, 0, COMM_GDFC2NEURON, MPI_COMM_WORLD);
	    	MPI_Send(h_gdpfc2bias,	fc2.pbias.size(), 	MPI_FLOAT, 0, COMM_GDFC2BIAS, 	MPI_COMM_WORLD);
	    }
	    if(rank == 0){
		//Recv rho(L-G) from every processor
	    	MPI_Recv(h_gdpconv1,	conv1.pconv.size(), 	MPI_FLOAT, i, COMM_GDCONV1,	MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	MPI_Recv(h_gdpconv1bias,conv1.pbias.size(), 	MPI_FLOAT, i, COMM_GDCONV1BIAS,	MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	MPI_Recv(h_gdpconv2,	conv2.pconv.size(), 	MPI_FLOAT, i, COMM_GDCONV2,	MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	MPI_Recv(h_gdpconv2bias,conv2.pbias.size(), 	MPI_FLOAT, i, COMM_GDCONV2BIAS,	MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	MPI_Recv(h_gdpfc1,	fc1.pneurons.size(),	MPI_FLOAT, i, COMM_GDFC1NEURON, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	MPI_Recv(h_gdpfc1bias,	fc1.pbias.size(),	MPI_FLOAT, i, COMM_GDFC1BIAS, 	MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	MPI_Recv(h_gdpfc2,	fc2.pneurons.size(), 	MPI_FLOAT, i, COMM_GDFC2NEURON, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	MPI_Recv(h_gdpfc2bias,	fc2.pbias.size(), 	MPI_FLOAT, i, COMM_GDFC2BIAS, 	MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	        //Copy rho(L-G) from device
                checkCudaErrors(cudaMemcpy(d_gdpconv1,		h_gdpconv1, sizeof(float) * conv1.pconv.size(), 	cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_gdpconv1bias, 	h_gdpconv1bias, sizeof(float) * conv1.pbias.size(), 	cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_gdpconv2,		h_gdpconv2, sizeof(float) * conv2.pconv.size(), 	cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_gdpconv2bias, 	h_gdpconv2bias, sizeof(float) * conv2.pbias.size(), 	cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_gdpfc1,		h_gdpfc1, sizeof(float) * fc1.pneurons.size(), 		cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_gdpfc1bias,   	h_gdpfc1bias, sizeof(float) * fc1.pbias.size(), 	cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_gdpfc2,		h_gdpfc2, sizeof(float) * fc2.pneurons.size(), 		cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_gdpfc2bias,   	h_gdpfc2bias, sizeof(float) * fc2.pbias.size(), 	cudaMemcpyHostToDevice));

                // Update weights
                context.UpdateGlobalWeights(learningRate, conv1, conv2,
                                  d_gpconv1, d_gpconv1bias, d_gpconv2, d_gpconv2bias, d_gpfc1, d_gpfc1bias, d_gpfc2, d_gpfc2bias,
                                  d_gdpconv1, d_gdpconv1bias, d_gdpconv2, d_gdpconv2bias, d_gdpfc1, d_gdpfc1bias, d_gdpfc2, d_gdpfc2bias);
	    }

	}

    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);
    
    if (FLAGS_save_data)
    {
        // Copy trained weights from GPU to CPU
        checkCudaErrors(cudaMemcpy(&conv1.pconv[0], d_pconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv1.pbias[0], d_pconv1bias, sizeof(float) * conv1.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv2.pconv[0], d_pconv2, sizeof(float) * conv2.pconv.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv2.pbias[0], d_pconv2bias, sizeof(float) * conv2.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc1.pneurons[0], d_pfc1, sizeof(float) * fc1.pneurons.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc1.pbias[0], d_pfc1bias, sizeof(float) * fc1.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc2.pneurons[0], d_pfc2, sizeof(float) * fc2.pneurons.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc2.pbias[0], d_pfc2bias, sizeof(float) * fc2.pbias.size(), cudaMemcpyDeviceToHost));
      
        // Now save data
        printf("Saving data to file\n");
        conv1.ToFile("conv1");
        conv2.ToFile("conv2");
        fc1.ToFile("ip1");
        fc2.ToFile("ip2");
    }
    

    float classification_error = 1.0f;

    int classifications = FLAGS_classify;
    if (classifications < 0)
        classifications = (int)test_size;
    
    // Test the resulting neural network's classification
    if (classifications > 0)
    {
        // Initialize a TrainingContext structure for testing (different batch size)
        TrainingContext test_context(FLAGS_gpu, 1, conv1, pool1, conv2, pool2, fc1, fc2);

        // Ensure correct workspaceSize is allocated for testing
        if (context.m_workspaceSize < test_context.m_workspaceSize)
        {
            checkCudaErrors(cudaFree(d_cudnn_workspace));
            checkCudaErrors(cudaMalloc(&d_cudnn_workspace, test_context.m_workspaceSize));
        }

        int num_errors = 0;
        for (int i = 0; i < classifications; ++i)
        {
            std::vector<float> data(width * height);
            // Normalize image to be in [0,1]
            for (int j = 0; j < width * height; ++j)
                data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;

            checkCudaErrors(cudaMemcpyAsync(d_data, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice));
            
            // Forward propagate test image
            test_context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                                            d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias,
                                            d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);

            // Perform classification
            std::vector<float> class_vec(10);

            // Copy back result
            checkCudaErrors(cudaMemcpy(&class_vec[0], d_fc2smax, sizeof(float) * 10, cudaMemcpyDeviceToHost));

            // Determine classification according to maximal response
            int chosen = 0;
            for (int id = 1; id < 10; ++id)
            {
                if (class_vec[chosen] < class_vec[id]) chosen = id;
            }

            if (chosen != test_labels[i])
                ++num_errors;
        }
        classification_error = (float)num_errors / (float)classifications;

        printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
    }
        
    // Free data structures
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_conv1));
    checkCudaErrors(cudaFree(d_pool1));
    checkCudaErrors(cudaFree(d_conv2));
    checkCudaErrors(cudaFree(d_pool2));
    checkCudaErrors(cudaFree(d_fc1));
    checkCudaErrors(cudaFree(d_fc2));
    checkCudaErrors(cudaFree(d_pconv1));
    checkCudaErrors(cudaFree(d_pconv1bias));
    checkCudaErrors(cudaFree(d_pconv2));
    checkCudaErrors(cudaFree(d_pconv2bias));
    checkCudaErrors(cudaFree(d_pfc1));
    checkCudaErrors(cudaFree(d_pfc1bias));
    checkCudaErrors(cudaFree(d_pfc2));
    checkCudaErrors(cudaFree(d_pfc2bias));
    checkCudaErrors(cudaFree(d_gconv1));
    checkCudaErrors(cudaFree(d_gconv1bias));
    checkCudaErrors(cudaFree(d_gconv2));
    checkCudaErrors(cudaFree(d_gconv2bias));
    checkCudaErrors(cudaFree(d_gfc1));
    checkCudaErrors(cudaFree(d_gfc1bias));
    checkCudaErrors(cudaFree(d_dfc1));
    checkCudaErrors(cudaFree(d_gfc2));
    checkCudaErrors(cudaFree(d_gfc2bias));
    checkCudaErrors(cudaFree(d_dfc2));
    checkCudaErrors(cudaFree(d_dpool1));
    checkCudaErrors(cudaFree(d_dconv2));
    checkCudaErrors(cudaFree(d_dpool2));    
    checkCudaErrors(cudaFree(d_labels));
    checkCudaErrors(cudaFree(d_dlossdata));
    checkCudaErrors(cudaFree(d_onevec));
    if (d_cudnn_workspace != nullptr)
        checkCudaErrors(cudaFree(d_cudnn_workspace));

    return 0;
}
