#pragma once

#include "util/common.h"
#include <cuda_runtime.h>

NT_HOST_DEVICE inline int3 make_int3(const float3 &v)
{
    return make_int3(int(v.x), int(v.y), int(v.z));
}

NT_HOST_DEVICE inline int3 operator-(const int3 &value, int scalar)
{
    return make_int3(value.x - scalar, value.y - scalar, value.z - scalar);
}

NT_HOST_DEVICE inline int3 operator<<(int value, const int3 &shift)
{
    return make_int3(value << shift.x, value << shift.y, value << shift.z);
}

NT_HOST_DEVICE inline int3 Max(const int3 &value, int floorValue)
{
    return make_int3(max(value.x, floorValue), max(value.y, floorValue), max(value.z, floorValue));
}
