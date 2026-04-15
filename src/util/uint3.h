#pragma once

#include "util/common.h"
#include <cstdint>
#include <cuda_runtime.h>

NT_HOST_DEVICE inline uint3 make_uint3(const float3 &v)
{
    return make_uint3(uint32_t(v.x), uint32_t(v.y), uint32_t(v.z));
}

NT_HOST_DEVICE inline uint3 operator<<(int value, const uint3 &shift)
{
    return make_uint3(value << shift.x, value << shift.y, value << shift.z);
}

NT_HOST_DEVICE inline uint3 operator%(const uint3 &value, uint32_t v)
{
    return make_uint3(value.x % v, value.y % v, value.z % v);
}
