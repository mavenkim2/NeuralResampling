#pragma once

#include "util/common.h"

NT_HOST_DEVICE inline float4 operator+(const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

NT_HOST_DEVICE inline float4 &operator+=(float4 &a, const float4 &b)
{
    a = a + b;
    return a;
}

NT_HOST_DEVICE inline float4 operator*(const float4 &a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
