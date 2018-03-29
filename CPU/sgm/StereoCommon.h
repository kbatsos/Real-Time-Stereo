#pragma once

#include "stdlib.h"
#include "stdint.h"
#include <algorithm>
#include <limits.h>

// Types
typedef float float32;
typedef double float64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef int8_t sint8;
typedef int16_t sint16;
typedef int32_t sint32;
// Defines

#define SINT32_MAX      ((sint32) 2147483647)

#define SINT16_MAX	    ((sint16)0x7fff)

#ifdef WIN32
    
    #ifndef NAN
        static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
        #define NAN (*(const float *) __nan)
    #endif
    #define FORCEINLINE __forceinline
    #define ALIGN16 __declspec(align(16))
    #define ALIGN32 __declspec(align(32))
    #define ASSERT(x) assert(x)
#else
    #define FORCEINLINE inline __attribute__((always_inline))
    #define ALIGN16 __attribute__ ((aligned(16)))
    #ifndef UINT16_MAX    
        #define UINT16_MAX    ((uint16)0xffffU)
    #endif
    #define ALIGN32 __attribute__ ((aligned(32)))
    #define ASSERT(x) assert(x)
#endif

#define UNUSED(x) (void)(x)

// Macros and inlines 
#ifdef MAX
    #undef MAX
#endif

#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#ifndef  MIN
    #define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define ROUND(x)    ( ((x) >= 0.0f) ? ((sint32)((x) + 0.5f)) : ((sint32)((x) - 0.5f)) )

#define SIGN(u,v)   ( ((v)>=0.0) ? ABS(u) : -ABS(u) )     
#define ABS(x) abs(x)

//inline float rangeCut(float min, float value, float max) {
//    if (value<min) 
//        value = min;
//    if (value>max)
//        value = max;
//    return value;
//}

// saturate casts
template<typename _Tp> static inline _Tp saturate_cast(uint8 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(sint8 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(uint16 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(sint16 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(uint32 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(sint32 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(float32 v) { return _Tp(v); }

template<> inline uint16 saturate_cast<uint16>(sint8 v)
{
    return (uint16)std::max((int)v, 0);
}
template<> inline uint16 saturate_cast<uint16>(sint16 v)
{
    return (uint16)std::max((int)v, 0);
}
template<> inline uint16 saturate_cast<uint16>(sint32 v)
{
    return (uint16)((unsigned)v <= (unsigned)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0);
}
template<> inline uint16 saturate_cast<uint16>(uint32 v)
{
    return (uint16)std::min(v, (unsigned)USHRT_MAX);
}

template<> inline uint16 saturate_cast<uint16>(float v)
{
    int iv = ROUND(v); return saturate_cast<uint16>(iv);
}

