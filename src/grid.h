#include "util/float3.h"
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>

namespace nrs
{

template <uint32_t numLevels>
struct Grid
{
    __half2 *hashTable;
    uint32_t hashTableSize;
    float b;
};

NT_DEVICE inline uint32_t SpatialHash(uint3 vertex)
{
    const int pi1 = 1;
    const int pi2 = 2654435761;
    const int pi3 = 805459861;

    uint32_t result = 0;
    result ^= vertex.x * pi1;
    result ^= vertex.y * pi2;
    result ^= vertex.z * pi3;

    return result;
}

NT_DEVICE inline float Quartic(float x, float invRadius)
{
    float u = x * invRadius;
    float tmp = fmaxf(1.0f - u * u, 0.0f);
    return (15.0f / 16.0f) * tmp * tmp;
}

NT_DEVICE inline float QuarticCDFDeriv(float x, float invRadius)
{
    return Quartic(x, invRadius) * invRadius;
}

NT_DEVICE inline float QuarticCDF(float x, float invRadius)
{
    float u = x * invRadius;
    float u2 = u * u;
    float u4 = u2 * u2;
    float v = (15.0f / 16.0f) * u * (1.0f - (2.0f / 3.0f) * u2 + (1.0f / 5.0f) * u4) + 0.5f;
    return Clamp(v, 0.0f, 1.0f);
}

template <uint32_t numBins>
NT_DEVICE inline void OneBlobEncoding(float s, __half *encoding)
{
    float invRadius = float(numBins);
    float leftCDF = QuarticCDF(-s, invRadius) /*+ QuarticCDF(-s - 1, invRadius) */ +
                    QuarticCDF(-s + 1, invRadius);

    NRS_UNROLL
    for (uint32_t bin = 0; bin < numBins; bin++)
    {
        float boundary = (bin + 1) / float(numBins);
        float rightCDF = QuarticCDF(boundary - s, invRadius) +
                         QuarticCDF(boundary - s - 1, invRadius) +
                         QuarticCDF(boundary - s + 1, invRadius);
        encoding[bin] = __half2float(rightCDF - leftCDF);
        leftCDF = rightCDF;
    }
}

template <uint32_t numLevels, uint32_t numBins>
__global__ void something(Grid<numLevels> grid, Bounds sceneBounds)
{
    // TODO: get samples somehow :)
    float3 position = make_float3(0.f);
    float3 direction = make_float3(0.f);

    float3 invHalfExtent = sceneBounds.max - sceneBounds.min;
    invHalfExtent.x = invHalfExtent.x == 0.f ? 0.f : 1.f / invHalfExtent.x;
    invHalfExtent.y = invHalfExtent.y == 0.f ? 0.f : 1.f / invHalfExtent.y;
    invHalfExtent.z = invHalfExtent.z == 0.f ? 0.f : 1.f / invHalfExtent.z;

    // TODO: I don't think you need to scale the position?
    // float3 relativeScenePosition = (position - sceneBounds.min) * invHalfExtent;

    __half featureVector[numLevels * 2 + 3 * numBins];
    NRS_UNROLL
    for (uint32_t level = 0; level < numLevels; level++)
    {
        float levelResolution = level * powf(grid.b, level);
        float3 gridPosition = position * levelResolution;
        int3 gridVertexLow = Floor(gridPosition);
        int3 gridVertexHigh = Ceil(gridPosition);
        float3 weight = gridPosition - Floor(gridPosition);

        float2 feature = 0.f;
        NRS_UNROLL
        for (int corner = 0; corner < 8; corner++)
        {
            int3 gridVertex;
            gridVertex.x = (corner & 1) ? gridVertexHigh.x : gridVertexLow.x;
            gridVertex.y = ((corner >> 1) & 1) ? gridVertexHigh.y : gridVertexLow.y;
            gridVertex.z = (corner >> 2) ? gridVertexHigh.z : gridVertexLow.z;

            float cornerWeight = (corner & 1) ? weight.x : 1 - weight.x;
            cornerWeight *= ((corner >> 1) & 1) ? weight.y : 1 - weight.y;
            cornerWeight *= (corner >> 2) ? weight.z : 1 - weight.z;

            const uint32_t hash = SpatialHash(gridVertex);
            const uint32_t hashTableIndex = lowHash % grid.hashTableSize;

            __half2 featureValue = hashTable[hashTableIndex];
            feature += __half22float2(featureValue) * cornerWeight;
        }
        featureVector[2 * level] = __float2half(feature.x);
        featureVector[2 * level + 1] = __float2half(feature.y);
    }
    OneBlobEncoding<numBins>(direction.x, featureVector + 2 * numLevels);
    OneBlobEncoding<numBins>(direction.y, featureVector + 2 * numLevels + numBins);
    OneBlobEncoding<numBins>(direction.z, featureVector + 2 * numLevels + 2 * numBins);
}

} // namespace nrs
