#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "openpgl/directional/neural/NeuralRadianceCacheHelpers.h"
#include "openpgl/directional/neural/NeuralRadianceCacheKernels.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_ASSERT(statement)                                                                                                              \
    do                                                                                                                                      \
    {                                                                                                                                       \
        cudaError_t result = (statement);                                                                                                   \
        if (result != cudaSuccess)                                                                                                          \
        {                                                                                                                                   \
            std::fprintf(stderr, "CUDA Error (%s): %s in %s (%s:%d)\n", cudaGetErrorName(result), cudaGetErrorString(result), #statement, \
                         __FILE__, __LINE__);                                                                                               \
            std::exit(EXIT_FAILURE);                                                                                                        \
        }                                                                                                                                   \
    } while (0)

namespace
{

constexpr uint32_t kTrainingGrid = 16;
constexpr uint32_t kNumSamples = kTrainingGrid * kTrainingGrid;
constexpr uint32_t kCellSize = 16;
constexpr uint32_t kImageWidth = kTrainingGrid * kCellSize;
constexpr uint32_t kImageHeight = kTrainingGrid * kCellSize;
constexpr uint32_t kNumLevels = OPENPGL_NEURAL_RADIANCE_CACHE_GRID_NUM_LEVELS;
constexpr uint32_t kFeaturesPerHashEntry = OPENPGL_NEURAL_RADIANCE_CACHE_GRID_NUM_FEATURES;
constexpr uint32_t kBaseResolution = 8;
constexpr uint32_t kMaxResolution = 2048;
constexpr uint32_t kMaxEntriesPerLevel = 1u << 16;
constexpr uint32_t kFeatureVectorSize = OPENPGL_NEURAL_RADIANCE_CACHE_FEATURE_VECTOR_SIZE;
constexpr uint32_t kAlignedFeatureVectorSize = ((kFeatureVectorSize + 15) / 16) * 16;
constexpr uint32_t kHiddenSize = OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE;
constexpr uint32_t kAlignedOutputSize = ((OPENPGL_NEURAL_RADIANCE_CACHE_OUTPUT_SIZE + 15) / 16) * 16;
constexpr uint32_t kLayer0Weights = kAlignedFeatureVectorSize * kHiddenSize;
constexpr uint32_t kHiddenWeights = kHiddenSize * kHiddenSize;
constexpr uint32_t kOutputWeights = kHiddenSize * kAlignedOutputSize;

struct OptimizerBuffer
{
    __half *values = nullptr;
    float *gradients = nullptr;
    float *firstMoments = nullptr;
    float *secondMoments = nullptr;
    uint32_t size = 0;
};

struct DeviceState
{
    openpgl::NeuralRadianceCacheSample *samples = nullptr;
    uint32_t *levelOffsets = nullptr;
    uint32_t hashTableEntries = 0;
    float levelScale = 1.0f;
    OptimizerBuffer hashTable;
    OptimizerBuffer layer0;
    OptimizerBuffer layer1;
    OptimizerBuffer layer2;
    OptimizerBuffer output;
};

__host__ __device__ inline float3 Normalize(float3 value)
{
    const float length = sqrtf(value.x * value.x + value.y * value.y + value.z * value.z);
    return length > 0.0f ? make_float3(value.x / length, value.y / length, value.z / length) : make_float3(0.0f, 1.0f, 0.0f);
}

__host__ __device__ inline float Dot3(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 AnalyticRadiance(float3 p, float3 w)
{
    const float3 sunDir = Normalize(make_float3(0.4f, 0.8f, 0.2f));
    const float sun = powf(fmaxf(Dot3(w, sunDir), 0.0f), 32.0f);
    const float spatial = 0.55f + 0.18f * sinf(12.0f * p.x + 3.0f * p.y) + 0.12f * cosf(10.0f * p.z);
    const float horizon = 0.5f + 0.5f * w.y;
    return make_float3(0.12f + spatial * (0.8f * sun + 0.2f * horizon), 0.12f + spatial * (0.4f * sun + 0.5f * horizon),
                       0.12f + spatial * (0.2f * sun + 0.9f * horizon));
}

__host__ __device__ inline void SampleAtIndex(uint32_t index, float3 &pos, float3 &dir)
{
    const uint32_t x = index % kTrainingGrid;
    const uint32_t y = index / kTrainingGrid;
    const float u = (float(x) + 0.5f) / float(kTrainingGrid);
    const float v = (float(y) + 0.5f) / float(kTrainingGrid);
    pos = make_float3(u, v, 0.5f + 0.25f * sinf(6.28318530718f * u) * cosf(6.28318530718f * v));

    const float phi = 6.28318530718f * u;
    const float theta = 3.14159265359f * v;
    dir = make_float3(sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi));
}

__global__ void GenerateSamplesKernel(openpgl::NeuralRadianceCacheSample *samples)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kNumSamples)
        return;

    float3 pos;
    float3 dir;
    SampleAtIndex(index, pos, dir);
    samples[index].pos = pos;
    samples[index].dir = dir;
    samples[index].radianceEstimate = AnalyticRadiance(pos, dir);
}

__device__ inline float3 EvaluateNrc(float3 pos, float3 dir, const __half *hashTable, const uint32_t *levelOffsets, float levelScale, const __half *weightsLayer0,
                                     const __half *weightsLayer1, const __half *weightsLayer2, const __half *weightsOutput)
{
    __half featureHalf[kFeatureVectorSize];
    openpgl::HashGridForward<__half>(featureHalf, pos, dir, hashTable, levelOffsets, kBaseResolution, levelScale);

    float input[kAlignedFeatureVectorSize] = {};
    float hidden0[kHiddenSize];
    float hidden1[kHiddenSize];
    float hidden2[kHiddenSize];

    for (uint32_t i = 0; i < kFeatureVectorSize; ++i)
        input[i] = __half2float(featureHalf[i]);

    for (uint32_t c = 0; c < kHiddenSize; ++c)
    {
        float sum = 0.0f;
        for (uint32_t r = 0; r < kAlignedFeatureVectorSize; ++r)
            sum += input[r] * __half2float(weightsLayer0[c * kAlignedFeatureVectorSize + r]);
        hidden0[c] = fmaxf(sum, 0.0f);
    }
    for (uint32_t c = 0; c < kHiddenSize; ++c)
    {
        float sum = 0.0f;
        for (uint32_t r = 0; r < kHiddenSize; ++r)
            sum += hidden0[r] * __half2float(weightsLayer1[c * kHiddenSize + r]);
        hidden1[c] = fmaxf(sum, 0.0f);
    }
    for (uint32_t c = 0; c < kHiddenSize; ++c)
    {
        float sum = 0.0f;
        for (uint32_t r = 0; r < kHiddenSize; ++r)
            sum += hidden1[r] * __half2float(weightsLayer2[c * kHiddenSize + r]);
        hidden2[c] = fmaxf(sum, 0.0f);
    }

    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    for (uint32_t r = 0; r < kHiddenSize; ++r)
    {
        result.x += hidden2[r] * __half2float(weightsOutput[0 * kHiddenSize + r]);
        result.y += hidden2[r] * __half2float(weightsOutput[1 * kHiddenSize + r]);
        result.z += hidden2[r] * __half2float(weightsOutput[2 * kHiddenSize + r]);
    }
    return result;
}

__device__ inline unsigned char ToByte(float value)
{
    value = value / (1.0f + value);
    value = fminf(fmaxf(value, 0.0f), 1.0f);
    return static_cast<unsigned char>(value * 255.0f + 0.5f);
}

__device__ inline uchar4 ToPixel(float3 color)
{
    return make_uchar4(ToByte(color.x), ToByte(color.y), ToByte(color.z), 255);
}

__device__ inline uchar4 ToErrorPixel(float3 prediction, float3 reference)
{
    const float ex = fabsf(prediction.x - reference.x) / fmaxf(reference.x, 1.0e-3f);
    const float ey = fabsf(prediction.y - reference.y) / fmaxf(reference.y, 1.0e-3f);
    const float ez = fabsf(prediction.z - reference.z) / fmaxf(reference.z, 1.0e-3f);
    const float e = fminf((ex + ey + ez) / 3.0f, 1.0f);
    return make_uchar4(static_cast<unsigned char>(255.0f * e), static_cast<unsigned char>(255.0f * (1.0f - fabsf(e - 0.5f) * 2.0f)),
                       static_cast<unsigned char>(255.0f * (1.0f - e)), 255);
}

enum class OutputMode : uint32_t
{
    Reference = 0,
    Prediction = 1,
    Error = 2,
};

__global__ void RenderOverfitImageKernel(uchar4 *pixels,
                                         OutputMode mode,
                                         const __half *hashTable,
                                         const uint32_t *levelOffsets,
                                         float levelScale,
                                         const __half *weightsLayer0,
                                         const __half *weightsLayer1,
                                         const __half *weightsLayer2,
                                         const __half *weightsOutput)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kImageWidth || y >= kImageHeight)
        return;

    const uint32_t sampleX = x / kCellSize;
    const uint32_t sampleY = y / kCellSize;
    const uint32_t sampleIndex = sampleY * kTrainingGrid + sampleX;

    float3 pos;
    float3 dir;
    SampleAtIndex(sampleIndex, pos, dir);
    const float3 reference = AnalyticRadiance(pos, dir);
    const float3 prediction = EvaluateNrc(pos, dir, hashTable, levelOffsets, levelScale, weightsLayer0, weightsLayer1, weightsLayer2, weightsOutput);

    uchar4 pixel;
    if (mode == OutputMode::Reference)
        pixel = ToPixel(reference);
    else if (mode == OutputMode::Prediction)
        pixel = ToPixel(prediction);
    else
        pixel = ToErrorPixel(prediction, reference);

    pixels[y * kImageWidth + x] = pixel;
}

void AllocateBuffer(OptimizerBuffer &buffer, uint32_t size, float scale, uint32_t seed, cudaStream_t stream)
{
    buffer.size = size;
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&buffer.values), sizeof(__half) * size));
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&buffer.gradients), sizeof(float) * size));
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&buffer.firstMoments), sizeof(float) * size));
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&buffer.secondMoments), sizeof(float) * size));
    CUDA_ASSERT(cudaMemsetAsync(buffer.gradients, 0, sizeof(float) * size, stream));
    CUDA_ASSERT(cudaMemsetAsync(buffer.firstMoments, 0, sizeof(float) * size, stream));
    CUDA_ASSERT(cudaMemsetAsync(buffer.secondMoments, 0, sizeof(float) * size, stream));
    CUDA_ASSERT(openpgl::LaunchInitializeHalfBufferKernel(buffer.values, size, scale, seed, stream));
}

void FreeBuffer(OptimizerBuffer &buffer)
{
    CUDA_ASSERT(cudaFree(buffer.values));
    CUDA_ASSERT(cudaFree(buffer.gradients));
    CUDA_ASSERT(cudaFree(buffer.firstMoments));
    CUDA_ASSERT(cudaFree(buffer.secondMoments));
    buffer = {};
}

std::vector<uint32_t> BuildLevelOffsets(float levelScale)
{
    std::vector<uint32_t> offsets(kNumLevels + 1, 0);
    for (uint32_t level = 0; level < kNumLevels; ++level)
    {
        const float scale = float(kBaseResolution) * std::pow(levelScale, float(level)) - 1.0f;
        const uint32_t resolution = uint32_t(std::ceil(scale)) + 1;
        const uint64_t dense = uint64_t(resolution) * uint64_t(resolution) * uint64_t(resolution);
        offsets[level + 1] = offsets[level] + uint32_t(std::min<uint64_t>(dense, kMaxEntriesPerLevel));
    }
    return offsets;
}

void ClearGradients(DeviceState &state, cudaStream_t stream)
{
    CUDA_ASSERT(cudaMemsetAsync(state.hashTable.gradients, 0, sizeof(float) * state.hashTable.size, stream));
    CUDA_ASSERT(cudaMemsetAsync(state.layer0.gradients, 0, sizeof(float) * state.layer0.size, stream));
    CUDA_ASSERT(cudaMemsetAsync(state.layer1.gradients, 0, sizeof(float) * state.layer1.size, stream));
    CUDA_ASSERT(cudaMemsetAsync(state.layer2.gradients, 0, sizeof(float) * state.layer2.size, stream));
    CUDA_ASSERT(cudaMemsetAsync(state.output.gradients, 0, sizeof(float) * state.output.size, stream));
}

void ApplyAdam(DeviceState &state, uint32_t step, cudaStream_t stream)
{
    constexpr float learningRate = 1.0e-3f;
    constexpr float weightDecay = 1.0e-5f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float epsilon = 1.0e-8f;
    const float beta1Power = std::pow(beta1, float(step));
    const float beta2Power = std::pow(beta2, float(step));
    const float invBatch = 1.0f / float(kNumSamples);

    CUDA_ASSERT(openpgl::LaunchAdamWNetworkWeightsKernel(state.layer0.values, state.layer0.gradients, state.layer0.firstMoments, state.layer0.secondMoments, state.layer0.size,
                                                        learningRate, beta1, beta2, beta1Power, beta2Power, epsilon, weightDecay, invBatch, stream));
    CUDA_ASSERT(openpgl::LaunchAdamWNetworkWeightsKernel(state.layer1.values, state.layer1.gradients, state.layer1.firstMoments, state.layer1.secondMoments, state.layer1.size,
                                                        learningRate, beta1, beta2, beta1Power, beta2Power, epsilon, weightDecay, invBatch, stream));
    CUDA_ASSERT(openpgl::LaunchAdamWNetworkWeightsKernel(state.layer2.values, state.layer2.gradients, state.layer2.firstMoments, state.layer2.secondMoments, state.layer2.size,
                                                        learningRate, beta1, beta2, beta1Power, beta2Power, epsilon, weightDecay, invBatch, stream));
    CUDA_ASSERT(openpgl::LaunchAdamWNetworkWeightsKernel(state.output.values, state.output.gradients, state.output.firstMoments, state.output.secondMoments, state.output.size,
                                                        learningRate, beta1, beta2, beta1Power, beta2Power, epsilon, weightDecay, invBatch, stream));
    CUDA_ASSERT(openpgl::LaunchAdamWHashFeaturesKernel(state.hashTable.values, state.hashTable.gradients, state.hashTable.firstMoments, state.hashTable.secondMoments,
                                                      state.hashTable.size, learningRate, beta1, beta2, beta1Power, beta2Power, epsilon, invBatch, stream));
}

std::string StripPngExtension(std::string path)
{
    if (path.size() >= 4)
    {
        const std::string suffix = path.substr(path.size() - 4);
        if (suffix == ".png" || suffix == ".PNG")
            path.resize(path.size() - 4);
    }
    return path;
}

bool WritePng(const std::string &path, const std::vector<uchar4> &pixels, uint32_t width, uint32_t height)
{
    return stbi_write_png(path.c_str(), int(width), int(height), 4, pixels.data(), int(width * sizeof(uchar4))) != 0;
}

} // namespace

int main(int argc, char **argv)
{
    uint32_t epochs = 2000;
    std::string outputStem = "nrc_overfit";
    if (argc > 1)
        epochs = uint32_t(std::max(1, std::atoi(argv[1])));
    if (argc > 2)
        outputStem = StripPngExtension(argv[2]);

    CUDA_ASSERT(cudaFree(nullptr));
    cudaStream_t stream = nullptr;
    CUDA_ASSERT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    DeviceState state;
    state.levelScale = std::exp(std::log(float(kMaxResolution) / float(kBaseResolution)) / float(kNumLevels - 1));
    const std::vector<uint32_t> levelOffsets = BuildLevelOffsets(state.levelScale);
    state.hashTableEntries = levelOffsets.back();
    const uint32_t hashValues = state.hashTableEntries * kFeaturesPerHashEntry;

    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&state.levelOffsets), sizeof(uint32_t) * levelOffsets.size()));
    CUDA_ASSERT(cudaMemcpyAsync(state.levelOffsets, levelOffsets.data(), sizeof(uint32_t) * levelOffsets.size(), cudaMemcpyHostToDevice, stream));
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&state.samples), sizeof(openpgl::NeuralRadianceCacheSample) * kNumSamples));

    AllocateBuffer(state.hashTable, hashValues, 1.0e-4f, 0x4f70674cU, stream);
    AllocateBuffer(state.layer0, kLayer0Weights, std::sqrt(2.0f / float(kAlignedFeatureVectorSize)), 0x1234U, stream);
    AllocateBuffer(state.layer1, kHiddenWeights, std::sqrt(2.0f / float(kHiddenSize)), 0x2345U, stream);
    AllocateBuffer(state.layer2, kHiddenWeights, std::sqrt(2.0f / float(kHiddenSize)), 0x3456U, stream);
    AllocateBuffer(state.output, kOutputWeights, std::sqrt(1.0f / float(kHiddenSize)), 0x4567U, stream);

    GenerateSamplesKernel<<<1, 256, 0, stream>>>(state.samples);
    CUDA_ASSERT(cudaGetLastError());

    openpgl::KernelParams params{};
    params.samples = state.samples;
    params.numSamples = kNumSamples;
    params.ringOffset = 0;
    params.ringSize = kNumSamples;
    params.hashTable = state.hashTable.values;
    params.hashTableGradients = state.hashTable.gradients;
    params.levelOffsets = state.levelOffsets;
    params.hashTableSize = state.hashTableEntries;
    params.baseResolution = kBaseResolution;
    params.levelScale = state.levelScale;
    params.weightsLayer0 = state.layer0.values;
    params.weightsLayer1 = state.layer1.values;
    params.weightsLayer2 = state.layer2.values;
    params.weightsOutputLayer = state.output.values;
    params.weightsGradientsLayer0 = state.layer0.gradients;
    params.weightsGradientsLayer1 = state.layer1.gradients;
    params.weightsGradientsLayer2 = state.layer2.gradients;
    params.weightsGradientsOutputLayer = state.output.gradients;

    for (uint32_t epoch = 0; epoch < epochs; ++epoch)
    {
        ClearGradients(state, stream);
        CUDA_ASSERT(openpgl::LaunchTrainingKernel(params, 2, stream));
        ApplyAdam(state, epoch + 1, stream);
    }

    uchar4 *devicePixels = nullptr;
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&devicePixels), sizeof(uchar4) * kImageWidth * kImageHeight));
    dim3 block(16, 16);
    dim3 grid((kImageWidth + block.x - 1) / block.x, (kImageHeight + block.y - 1) / block.y);
    std::vector<uchar4> pixels(kImageWidth * kImageHeight);

    auto renderAndWrite = [&](OutputMode mode, const std::string &suffix) {
        RenderOverfitImageKernel<<<grid, block, 0, stream>>>(devicePixels, mode, state.hashTable.values, state.levelOffsets, state.levelScale, state.layer0.values,
                                                             state.layer1.values, state.layer2.values, state.output.values);
        CUDA_ASSERT(cudaGetLastError());
        CUDA_ASSERT(cudaMemcpyAsync(pixels.data(), devicePixels, sizeof(uchar4) * pixels.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_ASSERT(cudaStreamSynchronize(stream));

        const std::string path = outputStem + "_" + suffix + ".png";
        if (!WritePng(path, pixels, kImageWidth, kImageHeight))
        {
            std::cerr << "Failed to write " << path << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    renderAndWrite(OutputMode::Reference, "reference");
    renderAndWrite(OutputMode::Prediction, "prediction");
    renderAndWrite(OutputMode::Error, "error");

    CUDA_ASSERT(cudaFree(devicePixels));
    FreeBuffer(state.hashTable);
    FreeBuffer(state.layer0);
    FreeBuffer(state.layer1);
    FreeBuffer(state.layer2);
    FreeBuffer(state.output);
    CUDA_ASSERT(cudaFree(state.samples));
    CUDA_ASSERT(cudaFree(state.levelOffsets));
    CUDA_ASSERT(cudaStreamDestroy(stream));

    std::cout << "Wrote " << outputStem << "_reference.png\n";
    std::cout << "Wrote " << outputStem << "_prediction.png\n";
    std::cout << "Wrote " << outputStem << "_error.png\n";
    std::cout << "after " << epochs << " epochs.\n";
    return EXIT_SUCCESS;
}
