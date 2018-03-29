// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details

#pragma once

#include "StereoCommon.h"
#include <smmintrin.h> // intrinsics
#include <emmintrin.h>
#include <nmmintrin.h>

#include <iostream>
#include <fstream>

#define HW_POPCNT

/* hamming costs and population counts */
extern uint16 m_popcount16LUT[UINT16_MAX+1];

void fillPopCount16LUT();

#ifdef HW_POPCNT
    #define POPCOUNT32 _mm_popcnt_u32
    #if defined(_M_X64) || defined(__amd64__) || defined(__amd64)
        #define POPCOUNT64 (uint16)_mm_popcnt_u64
    #else 
        #define POPCOUNT64 popcount64LUT
    #endif
#else
    #define POPCOUNT32 popcount32
    #define POPCOUNT64 popcount64LUT
#endif

inline uint16 popcount32(const uint32& i)
{
    return (m_popcount16LUT[i&0xFFFF] + m_popcount16LUT[i>>16]);
}

// pop count for 4 32bit values
FORCEINLINE __m128i popcount32_4(const __m128i& a)
{
    __m128i b = _mm_srli_epi16(a,4); // psrlw       $4, %%xmm1

    ALIGN16 const uint32 _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
    const __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);

    const __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F); //_mm_set1_epi8(0xf);

    __m128i a2 = _mm_and_si128(a, xmm6); // pand    %%xmm6, %%xmm0  ; xmm0 - lower nibbles
    b = _mm_and_si128(b, xmm6); // pand    %%xmm6, %%xmm1  ; xmm1 - higher nibbles

    __m128i popA = _mm_shuffle_epi8(xmm7,a2); // pshufb  %%xmm0, %%xmm2  ; xmm2 = vector of popcount for lower nibbles
    __m128i popB = _mm_shuffle_epi8(xmm7,b); //  pshufb  %%xmm1, %%xmm3  ; xmm3 = vector of popcount for higher nibbles

    __m128i popByte = _mm_add_epi8(popA, popB); // paddb   %%xmm3, %%xmm2  ; xmm2 += xmm3 -- vector of popcount for bytes;
    
    // How to get to added quadwords?
    const __m128i ZERO = _mm_setzero_si128();

    // with horizontal adds

    __m128i upper = _mm_unpackhi_epi8(popByte, ZERO);
    __m128i lower = _mm_unpacklo_epi8(popByte, ZERO);
    __m128i popUInt16 = _mm_hadd_epi16(lower,upper); // uint16 pop count
    __m128i popUInt32 = _mm_hadd_epi16(popUInt16,ZERO); // uint32 pop count
    
    return popUInt32;
}

// pop count for 4 32bit values
FORCEINLINE __m128i popcount32_4(const __m128i& a, const __m128i& lut,const __m128i& mask)
{

    __m128i b = _mm_srli_epi16(a,4); // psrlw       $4, %%xmm1

    __m128i a2 = _mm_and_si128(a, mask); // pand    %%xmm6, %%xmm0  ; xmm0 - lower nibbles
    b = _mm_and_si128(b, mask); // pand    %%xmm6, %%xmm1  ; xmm1 - higher nibbles

    __m128i popA = _mm_shuffle_epi8(lut,a2); // pshufb  %%xmm0, %%xmm2  ; xmm2 = vector of popcount for lower nibbles
    __m128i popB = _mm_shuffle_epi8(lut,b); //  pshufb  %%xmm1, %%xmm3  ; xmm3 = vector of popcount for higher nibbles

    __m128i popByte = _mm_add_epi8(popA, popB); // paddb   %%xmm3, %%xmm2  ; xmm2 += xmm3 -- vector of popcount for bytes;

    // How to get to added quadwords?
    const __m128i ZERO = _mm_setzero_si128();

    // Version 1 - with horizontal adds

    __m128i upper = _mm_unpackhi_epi8(popByte, ZERO);
    __m128i lower = _mm_unpacklo_epi8(popByte, ZERO);
    __m128i popUInt16 = _mm_hadd_epi16(lower,upper); // uint16 pop count
    // the lower four 16 bit values contain the uint32 pop count
    __m128i popUInt32 = _mm_hadd_epi16(popUInt16,ZERO);

    return popUInt32;
}

// unsigned to 32bit
inline uint16 hamDist32(const uint32& x, const uint32& y)
{
    uint16 dist = 0, val = (uint16)(x ^ y);

    // Count the number of set bits
    while(val)
    {
        ++dist; 
        val &= val - 1;
    }

    return dist;
}

//uint16 hamDist64(uint64 x, uint64 y);

const uint64 m1  = 0x5555555555555555; //binary: 0101...
const uint64 m2  = 0x3333333333333333; //binary: 00110011..
const uint64 m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
const uint64 m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
const uint64 m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
const uint64 m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
const uint64 hff = 0xffffffffffffffff; //binary: all ones
const uint64 h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...

inline uint16 popcount64(uint64 x) {
    x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
    x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits 
    x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits 
    return (x * h01)>>56;  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
}

inline uint16 popcount64LUT(const uint64& i)
{
    return (m_popcount16LUT[i&0xFFFF] + m_popcount16LUT[(i>>16) & 0xFFFF] 
    + m_popcount16LUT[(i>>32) & 0xFFFF]) + m_popcount16LUT[i>>48];
}

inline uint16* getDispAddr_xyd(uint16* dsi, sint32 width, sint32 disp, sint32 i, sint32 j, sint32 k)
{
    return dsi + i*(disp*width) + j*disp + k;
}

/* fill disparity cube */
void costMeasureCensus5x5_xyd_SSE(uint32* intermediate1, uint32* intermediate2, 
    const sint32 height, const sint32 width, const sint32 dispCount, const uint16 invalidDispValue, uint16* dsi, sint32 numThreads);
void costMeasureCensusCompressed5x5_xyd_SSE(uint32* intermediate1, uint32* intermediate2,
    sint32 height, sint32 width, sint32 dispCount, const uint16 invalidDispValue, sint32 dispSubSample, uint16* dsi, sint32 numThreads);

void costMeasureCensus9x7_xyd_parallel(uint64* intermediate1, uint64* intermediate2,int height, int width, int dispCount, uint16* dsi,
    sint32 numThreads);
void costMeasureCensusCompressed9x7_xyd(uint64* intermediate1, uint64* intermediate2,
    sint32 height, sint32 width, sint32 dispCount, sint32 dispSubSample, uint16* dsi);


/* WTA disparity selection in disparity cube */
void matchWTA_SSE(float32* dispImg, uint16* &dsiAgg, const sint32 width, const sint32 height, 
    const sint32 maxDisp, const float32 uniqueness);
void matchWTAAndSubPixel_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness);

void matchWTARight_SSE(float32* dispImg, uint16* &dsiAgg, const sint32 width, const sint32 height, 
    const sint32 maxDisp, const float32 uniqueness);

void doLRCheck(float32* dispImg, float32* dispCheckImg,const sint32 width, const sint32 height, const sint32 lrThreshold);
void doRLCheck(float32* dispRightImg, float32* dispCheckImg,const sint32 width, const sint32 height, const sint32 lrThreshold);

void subPixelRefine(float32* dispImg, uint16* dsiImg,
    const sint32 width, const sint32 height, const sint32 maxDisp, sint32 method);

void uncompressDisparities_SSE(float32* dispImg, const sint32 width, const sint32 height, uint32 stepwidth);
