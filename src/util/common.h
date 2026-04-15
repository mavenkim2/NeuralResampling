#pragma once

#if defined(__CUDACC__)
#define NT_HOST_DEVICE __host__ __device__
#define NT_DEVICE __device__
#else
#define NT_HOST_DEVICE
#define NT_DEVICE
#endif

#define NRS_UNROLL _Pragma("unroll")

NT_HOST_DEVICE inline constexpr int AlignUp(int val, int pow2)
{
    return (val + pow2 - 1) & ~(pow2 - 1);
}

template <typename T>
NT_HOST_DEVICE inline T Clamp(T v, T low, T high)
{
    return v < low ? low : (v > high ? high : v);
}
