#pragma once

#include "util/common.h"
#include <cmath>
#include <cuda_runtime.h>

NT_HOST_DEVICE inline float2 operator+(const float2 &a, const float2 &b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

NT_HOST_DEVICE inline float2 operator+(const float2 &a, float b)
{
    return make_float2(a.x + b, a.y + b);
}

NT_HOST_DEVICE inline float2 &operator+=(float2 &a, const float2 &b)
{
    a = a + b;
    return a;
}

NT_HOST_DEVICE inline float2 operator-(const float2 &a, const float2 &b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

NT_HOST_DEVICE inline float2 operator-(const float2 &a, float b)
{
    return make_float2(a.x - b, a.y - b);
}

NT_HOST_DEVICE inline float2 operator*(const float2 &a, const float2 &b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

NT_HOST_DEVICE inline float2 operator*(const float2 &a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

NT_HOST_DEVICE inline float2 operator*(float a, const float2 &b)
{
    return b * a;
}

NT_HOST_DEVICE inline float2 operator/(const float2 &a, float b)
{
    return make_float2(a.x / b, a.y / b);
}

NT_HOST_DEVICE inline float2 operator/(const float2 &a, const float2 &b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}

NT_HOST_DEVICE inline float2 Floor(const float2 &value)
{
    return make_float2(floorf(value.x), floorf(value.y));
}

