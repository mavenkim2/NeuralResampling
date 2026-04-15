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

template <uint32_t numLevels>
__global__ void something(Grid<numLevels> grid, Bounds sceneBounds)
{
    // TODO: get samples somehow :)
    float3 position = make_float3(0.f);

    float3 invHalfExtent = sceneBounds.max - sceneBounds.min;
    invHalfExtent.x = invHalfExtent.x == 0.f ? 0.f : 1.f / invHalfExtent.x;
    invHalfExtent.y = invHalfExtent.y == 0.f ? 0.f : 1.f / invHalfExtent.y;
    invHalfExtent.z = invHalfExtent.z == 0.f ? 0.f : 1.f / invHalfExtent.z;

    // TODO: do you need the scene bounds?
    float3 relativeScenePosition = (position - sceneBounds.min) * invHalfExtent;

    __half featureVector[numLevels * 2];
    NRS_UNROLL
    for (uint32_t level = 0; level < numLevels; level++)
    {
        float levelResolution = level * powf(grid.b, level);
        float3 gridPosition = relativeScenePosition * levelResolution;
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
}

} // namespace nrs
