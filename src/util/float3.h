#pragma once

#include "util/common.h"
#include "util/int3.h"
#include <cmath>
#include <cuda_runtime.h>

NT_HOST_DEVICE inline float3 make_float3(float f)
{
    return make_float3(f, f, f);
}

NT_HOST_DEVICE inline float3 make_float3(const int3 &a)
{
    return make_float3((float)a.x, (float)a.y, (float)a.z);
}

NT_HOST_DEVICE inline float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

NT_HOST_DEVICE inline float3 operator+(const float3 &a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

NT_HOST_DEVICE inline float3 &operator+=(float3 &a, const float3 &b)
{
    a = a + b;
    return a;
}

NT_HOST_DEVICE inline float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

NT_HOST_DEVICE inline float3 operator-(const float3 &a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

NT_HOST_DEVICE inline float3 operator*(const float3 &a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

NT_HOST_DEVICE inline float3 operator*(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

NT_HOST_DEVICE inline float3 operator*(float a, const float3 &b)
{
    return b * a;
}

NT_HOST_DEVICE inline float3 operator/(const float3 &a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

NT_HOST_DEVICE inline float Dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

NT_HOST_DEVICE inline float3 Floor(const float3 &value)
{
    return make_float3(floorf(value.x), floorf(value.y), floorf(value.z));
}

NT_HOST_DEVICE inline float3 Ceil(const float3 &value)
{
    return make_float3(ceilf(value.x), ceilf(value.y), ceilf(value.z));
}

NT_HOST_DEVICE inline float3 Max(const float3 &value, float floorValue)
{
    return make_float3(
        fmaxf(value.x, floorValue), fmaxf(value.y, floorValue), fmaxf(value.z, floorValue));
}

NT_HOST_DEVICE inline float3 Ldexp(const float3 &value, int exponent)
{
    return make_float3(
        ldexpf(value.x, exponent), ldexpf(value.y, exponent), ldexpf(value.z, exponent));
}

NT_HOST_DEVICE inline float3 Ldexp(const float3 &value, const int3 &exponent)
{
    return make_float3(
        ldexpf(value.x, exponent.x), ldexpf(value.y, exponent.y), ldexpf(value.z, exponent.z));
}
