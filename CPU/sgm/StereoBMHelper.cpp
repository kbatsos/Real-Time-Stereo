// Copyright � Robert Spangenberg, 2014.
// See license.txt for more details

#include "StereoCommon.h"
#include "StereoBMHelper.h"

#include <assert.h>
#include <cmath>
#include <limits>
#include <smmintrin.h> // intrinsics
#include <emmintrin.h>
#include <nmmintrin.h>
#include <string.h>
#include <cstdlib>

//#define USE_AVX2

// pop count LUT for for uint16
uint16 m_popcount16LUT[UINT16_MAX+1];

void fillPopCount16LUT()
{
    // popCount LUT
    for (int i=0; i < UINT16_MAX+1; i++) {
        m_popcount16LUT[i] = hamDist32(i,0);
    }
}

void costMeasureCensus5x5Line_xyd_SSE(uint32* intermediate1, uint32* intermediate2
    ,const int width,const int dispCount, const uint16 invalidDispValue, uint16* dsi, const int lineStart,const int lineEnd)
{
    ALIGN16 const unsigned _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
    const __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);
    const __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F);

    for (sint32 i=lineStart;i < lineEnd;i++) {
        uint32* pBase = intermediate1+i*width;
        uint32* pMatchRow = intermediate2+i*width;
        for (uint32 j=0; j < (uint32)width; j++) {
            uint32* pBaseJ = pBase + j;
            uint32* pMatchRowJmD = pMatchRow + j - dispCount +1;

            int d=dispCount - 1;
            
            for (; d >(sint32)j && d >= 0;d--) {
                *getDispAddr_xyd(dsi, width, dispCount, i, j, d) = invalidDispValue;
                pMatchRowJmD++;
            }

            int dShift4m1 = ((d-1) >> 2)*4;
            int diff = d - dShift4m1;
            // rest
            if (diff != 0) {
                for (; diff >= 0 && d >= 0;d--,diff--) {
                    uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
                    *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = cost;
                    pMatchRowJmD++;
                }
            }

            // 4 costs at once
            __m128i lPoint4 = _mm_set1_epi32(*pBaseJ);
            d -= 3;

            uint16* baseAddr = getDispAddr_xyd(dsi,width, dispCount, i,j,0);
            for (; d >= 0;d-=4) {
                // flip the values
                __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), 0x1b); //mask = 00 01 10 11
                _mm_storel_pi((__m64*)(baseAddr+d), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4),xmm7,xmm6)));
                pMatchRowJmD+=4;
            }
            
        }
    }
} 

void costMeasureCensus5x5_xyd_SSE(uint32* intermediate1, uint32* intermediate2
    , const int height,const int width, const int dispCount, const uint16 invalidDispValue, uint16* dsi,
    sint32 numThreads)
{
    // first 2 lines are empty
    for (int i=0;i<2;i++) {
        for (int j=0; j < width; j++) {
            for (int d=0; d <= dispCount-1;d++) {
                *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = invalidDispValue;
            }
        }
    }

    if (numThreads != 4) 
    {
#pragma omp parallel num_threads(2)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dsi, 2, height/2);
                }
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dsi, height/2, height-2);
                }
            }
        }
    } else if (numThreads == 4) {
#pragma omp parallel num_threads(4)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dsi, 2, height/4);
                }
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dsi, height/4, height/2);
                }
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dsi, height/2, height-height/4);
                }
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dsi, height-height/4, height-2);
                }
            }
        }
    }
    /* last 2 lines are empty*/
    for (int i=height-2;i<height;i++) {
        for (int j=0; j < width; j++) {
            for (int d=0; d <= dispCount-1;d++) {
                *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = invalidDispValue;
            }
        }
    }
}

void costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(uint32* intermediate1, uint32* intermediate2,const int width,
    const int dispCount, const uint16 invalidDispValue, const sint32 dispCountCompressed, const sint32 dispSubSample, const sint32 dispCountLow,
    uint16* dsi, const sint32 lineStart,const sint32 lineEnd)
{
    for (sint32 i=lineStart;i < lineEnd;i++) {
        uint32* pBase = intermediate1+i*width;
        uint32* pMatchRow = intermediate2+i*width;
        for (uint32 j=0; j < (uint32)width; j++) {
            uint32* pBaseJ = pBase + j;
            uint32* pMatchRowJmD = pMatchRow + j - dispCount +1;

            sint32 d=dispCount - 1;
            
            for (; d >(sint32)j && d >= 0 && d>=dispCountLow; d--) {
                sint32 compressedD = dispCountLow+(d-dispCountLow)/dispSubSample;
                *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD) = invalidDispValue;
                pMatchRowJmD++;
            }

            for (; d >(sint32)j && d >= 0; d--) {
                *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = invalidDispValue;
                pMatchRowJmD++;
            }

            // fill valid disparities

            if (d > dispCountLow - 1) {
                // align to sub-sampled disparities
                
                // get multiple of sub-sampling
                sint32 shift = (d-dispCountLow+1) % 8;
                
                // rest
                if (shift > 0) {
                    sint32 modRes = d % dispSubSample;
                    d -= modRes;
                    pMatchRowJmD+= modRes;
                    shift = (shift+1) / 2;

                    // advance through sub-sampled disparities
                    for (; shift > 0 && d >= dispCountLow;d-= dispSubSample, shift-=1) {
                        uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
                        sint32 compressedD = dispCountLow+(d-dispCountLow)/dispSubSample;
                        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD) = cost;
                        pMatchRowJmD += dispSubSample;
                    }
                    d++;
                    pMatchRowJmD -= 1;
                }
            } else {
                // align to full-sampled disparities
                sint32 dShift4m1 = ((d-1) >> 2)*4;
                sint32 diff = d - dShift4m1;
                // rest
                if (diff > 0) {
                    // advance through full-sampled disparities
                    for (; diff >= 0 && d >= 0;d--,diff--) {
                        uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
                        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = cost;
                        pMatchRowJmD++;
                    }
                }
            }

            // 4 costs at once
            ALIGN16 const unsigned _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
            __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);
            __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F);

            __m128i lPoint4 = _mm_set1_epi32(*pBaseJ);
            

            // sub-sampled disparities
            for (; d >= dispCountLow;d-=8) {
                // flip the values
                __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), _MM_SHUFFLE(1,3,1,3)); //mask = 00 01 10 11
                __m128i rPoint4n = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)(pMatchRowJmD+4)), _MM_SHUFFLE(1,3,1,3)); //mask = 00 01 10 11
                const sint32 mask = 1<<0 | 1<<1 | 1<<2 | 1<<3;
                __m128i rPoint4Sub = _mm_blend_epi16(rPoint4, rPoint4n, mask); // first and second of first, third and fourth of second
                sint32 compressedD = dispCountLow+(d-7-dispCountLow)/dispSubSample;
                _mm_storel_pi((__m64*)getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4Sub),xmm7,xmm6)));
                pMatchRowJmD+=8;
            }

            // full sampled disparities
            d -=3;
            for (; d >= 0;d-=4) {
                // flip the values
                __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), 0x1b); //mask = 00 01 10 11
                _mm_storel_pi((__m64*)getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4),xmm7,xmm6)));
                pMatchRowJmD+=4;
            }
        }
    }
}

void costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(uint32* intermediate1, uint32* intermediate2,const int width,
    const int dispCount, const uint16 invalidDispValue, const sint32 dispCountCompressed, const sint32 dispSubSample, const sint32 dispCountLow,
    uint16* dsi, const sint32 lineStart,const sint32 lineEnd)
{
    for (sint32 i=lineStart;i < lineEnd;i++) {
        uint32* pBase = intermediate1+i*width;
        uint32* pMatchRow = intermediate2+i*width;
        for (uint32 j=0; j < (uint32)width; j++) {
            uint32* pBaseJ = pBase + j;
            uint32* pMatchRowJmD = pMatchRow + j - dispCount +1;

            sint32 d=dispCount - 1;

            for (; d >(sint32)j && d >= 0 && d>=dispCountLow; d--) {
                sint32 compressedD = dispCountLow+(d-dispCountLow)/dispSubSample;
                *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD) = invalidDispValue;
                pMatchRowJmD++;
            }

            for (; d >(sint32)j && d >= 0; d--) {
                *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = invalidDispValue;
                pMatchRowJmD++;
            }

            // fill valid disparities

            if (d > dispCountLow - 1) {
                // align to sub-sampled disparities

                // get multiple of sub-sampling times 4
                sint32 shift = (d-dispCountLow+1) % 16;

                // rest
                if (shift > 0) {
                    sint32 modRes = d % dispSubSample;
                    d -= modRes;
                    pMatchRowJmD+= modRes;
                    shift -= modRes;

                    // advance through sub-sampled disparities
                    for (; shift > 0 && d >= dispCountLow;d-= dispSubSample, shift-=4) {
                        uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
                        sint32 compressedD = dispCountLow+(d-dispCountLow)/dispSubSample;
                        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD) = cost;
                        pMatchRowJmD += dispSubSample;
                    }
                    d+=3;
                    pMatchRowJmD -= 3;
                }
            } else {
                // align to full-sampled disparities
                sint32 dShift4m1 = ((d-1) >> 2)*4;
                sint32 diff = d - dShift4m1;
                // rest
                if (diff > 0) {
                    // advance through full-sampled disparities
                    for (; diff >= 0 && d >= 0;d--,diff--) {
                        uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
                        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = cost;
                        pMatchRowJmD++;
                    }
                }
            }

            // 4 costs at once
            ALIGN16 const unsigned _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
            __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);
            __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F);

            __m128i lPoint4 = _mm_set1_epi32(*pBaseJ);


            // sub-sampled disparities
            bool entered = false;
            for (; d >= dispCountLow;d-=16) {
                // flip the values
                __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), _MM_SHUFFLE(3,1,2,0)); //mask = 00 01 10 11
                __m128i rPoint4n = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)(pMatchRowJmD+4)), _MM_SHUFFLE(2,3,1,0)); //mask = 00 01 10 11
                __m128i rPoint4n2 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)(pMatchRowJmD+8)), _MM_SHUFFLE(1,2,3,0)); //mask = 00 01 10 11
                __m128i rPoint4n3 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)(pMatchRowJmD+12)), _MM_SHUFFLE(0,1,2,3)); //mask = 00 01 10 11

                const sint32 mask = 1<<0 | 1<<1 | 1<<4 | 1<<5;
                // first of first, second of second, third of third, fourth of fourth
                __m128i rPoint4Sub1 = _mm_blend_epi16(rPoint4n2, rPoint4n3, mask); 
                __m128i rPoint4Sub2 = _mm_blend_epi16(rPoint4, rPoint4n, mask);
                const sint32 mask2 = 1<<0 | 1<<1 | 1<<2 | 1<<3;
                __m128i rPoint4Sub = _mm_blend_epi16(rPoint4Sub2, rPoint4Sub1, mask2); 

                sint32 compressedD = dispCountLow+(d-15-dispCountLow)/dispSubSample;
                _mm_storel_pi((__m64*)getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4Sub),xmm7,xmm6)));
                pMatchRowJmD+=16;
                entered = true;
            }
            if (entered) {
                d = 63;
            }
            // full sampled disparities
            d -=3;
            for (; d >= 0;d-=4) {
                // flip the values
                __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), 0x1b); //mask = 00 01 10 11
                _mm_storel_pi((__m64*)getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4),xmm7,xmm6)));
                pMatchRowJmD+=4;
            }
        }
    }
} 

void costMeasureCensusCompressed5x5_xyd_SSE(uint32* intermediate1, uint32* intermediate2
    , sint32 height, sint32 width, sint32 dispCount, const uint16 invalidDispValue, sint32 dispSubSample, uint16* dsi, sint32 numThreads)
{
    sint32 dispCountCompressed=64;
    sint32 dispCountLow = 64;

    if (dispCount > 64) {
        if (dispCount == 128) {
            dispCountCompressed = 64+(dispCount-64)/dispSubSample; // sample every dispSubSample disparity
            dispCountLow = 64;
        } else {
            assert(false);
        }
    } else {
        dispCountCompressed = 64;
        dispCountLow = 64;
    }

    // first 2 lines are empty
    for (int i=0;i<2;i++) {
        for (int j=0; j < width; j++) {
            for (int d=0; d <= dispCountCompressed-1;d++) {
                *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = invalidDispValue;
            }
        }
    }

    if (dispSubSample == 2) {

        if (numThreads != 4) 
        {
#pragma omp parallel num_threads(2)
            {
#pragma omp sections nowait
                {
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, 2, height/2);
                    }
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/2, height-2);
                    }
                }
            }
        } else if (numThreads == 4) {
#pragma omp parallel num_threads(4)
            {
#pragma omp sections nowait
                {
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, 2, height/4);
                    }
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/4, height/2);
                    }
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/2, height-height/4);
                    }
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height-height/4, height-2);
                    }
                }
            }
        }
    } else if (dispSubSample == 4) {
        if (numThreads != 4) 
        {
#pragma omp parallel num_threads(2)
            {
#pragma omp sections nowait
                {
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, 2, height/2);
                    }
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/2, height-2);
                    }
                }
            }
        } else if (numThreads == 4) {
#pragma omp parallel num_threads(4)
            {
#pragma omp sections nowait
                {
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, 2, height/4);
                    }
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/4, height/2);
                    }
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/2, height-height/4);
                    }
#pragma omp section
                    {
                        costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height-height/4, height-2);
                    }
                }
            }
        }

    }
    /* last 2 lines are empty*/
    for (int i=height-2;i<height;i++) {
        for (int j=0; j < width; j++) {
            for (int d=0; d <= dispCountCompressed-1;d++) {
                *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = invalidDispValue;
            }
        }
    }
}

void costMeasureCensus9x7Line_xyd(uint64* intermediate1, uint64* intermediate2,int width, int dispCount, uint16* dsi, int startLine, int endLine)
{
    // content
    for (int i=startLine;i < endLine;i++) {
        uint64* baseRow = intermediate1+ width*i;
        uint64* matchRow = intermediate2+width*i;

        for (int j=0;j < width;j++) {
            uint64* pBaseJ = baseRow + j;
            uint64* pMatchRowJmD = matchRow + j - dispCount +1;

            sint32 d = dispCount -1;
            for (; d >(sint32)j && d >= 0;d--) {
                *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = 255;
                pMatchRowJmD++;
            }
            while (d >= 0) {
                uint16 cost = POPCOUNT64(*pBaseJ ^ *pMatchRowJmD);
                pMatchRowJmD++;
                *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = cost;
                d--;
            }
        }
    }
}


void costMeasureCensus9x7_xyd_parallel(uint64* intermediate1, uint64* intermediate2,int height, int width, int dispCount, uint16* dsi,
    sint32 numThreads)
{
    // first 3 lines are empty
    for (int i=0;i<3;i++) {
        for (int j=0; j < width; j++) {
            for (int d=0; d <= dispCount-1;d++) {
                *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = 255;
            }
        }
    }

    if (numThreads != 4) 
    {
#pragma omp parallel /*shared(dsi,height, width, dispCount, baseImg, matchImg),*/ num_threads(2)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    costMeasureCensus9x7Line_xyd(intermediate1, intermediate2, width, dispCount, dsi, 2, height/2);
                }
#pragma omp section
                {
                    costMeasureCensus9x7Line_xyd(intermediate1, intermediate2,width, dispCount, dsi, height/2, height-2);
                }
            }
        }
    } else if (numThreads == 4) {
#pragma omp parallel /*shared(dsi,height, width, dispCount, baseImg, matchImg),*/ num_threads(4)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    costMeasureCensus9x7Line_xyd(intermediate1, intermediate2, width, dispCount, dsi, 2, height/4);
                }
#pragma omp section
                {
                    costMeasureCensus9x7Line_xyd(intermediate1, intermediate2,width, dispCount, dsi, height/4, height/2);
                }
#pragma omp section
                {
                    costMeasureCensus9x7Line_xyd(intermediate1, intermediate2,width, dispCount, dsi, height/2, height-height/4);
                }
#pragma omp section
                {
                    costMeasureCensus9x7Line_xyd(intermediate1, intermediate2,width, dispCount, dsi, height-height/4, height-2);
                }
            }
        }
    }
    /* last 3 lines are empty*/
    for (int i=height-3;i<height;i++) {
        for (int j=0; j < width; j++) {
            for (int d=0; d <= dispCount-1;d++) {
                *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = 255;
            }
        }
    }
}

void costMeasureCensusCompressed9x7_xyd(uint64* intermediate1, uint64* intermediate2
    , sint32 height, sint32 width, sint32 dispCount, sint32 dispSubSample, uint16* dsi)
{
    sint32 dispCountCompressed = dispCount;
    sint32 dispCountLow = dispCount;
    if (dispCount > 64) {
        if (dispCount == 128) {
            dispCountLow = 64;
            if (dispSubSample == 2) {
                dispCountCompressed = 96;
            }
            else if (dispSubSample == 4) {
                dispCountCompressed = 80;
            }
            else {
                assert(false);
            }

        }
        else {
            assert(false);
        }

    }
    // first 3 lines are empty
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < width; j++) {
            for (int d = 0; d < dispCountCompressed; d++) {
                *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = 255;
            }
        }
    }

    // content
    for (int i = 3; i < height - 3; i++) {
        uint64* baseRow = intermediate1 + width*i;
        uint64* matchRow = intermediate2 + width*i;

        // stepSize 1
        for (int d = 0; d < dispCountLow; d++) {
            uint64* pBaseJ = baseRow + d;
            uint64* pMatchRowJmD = matchRow;
            for (int j = 0; j < d; j++) {
                *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = 255;
            }
            for (int j = d; j < width; j++) {
                uint16 cost = POPCOUNT64(*pBaseJ ^ *pMatchRowJmD);
                pBaseJ++; pMatchRowJmD++;
                *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = cost;
            }
        }

        // stepSize dispSubSample
        for (int d = dispCountLow; d < dispCountCompressed; d++) {
            sint32 realDisp = dispCountLow + (d - dispCountLow)*dispSubSample;
            uint64* pBaseJ = baseRow + realDisp;
            uint64* pMatchRowJmD = matchRow;
            for (int j = 0; j < realDisp; j++) {
                *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = 255;
            }
            for (int j = realDisp; j < width; j++) {
                uint16 cost = POPCOUNT64(*pBaseJ ^ *pMatchRowJmD);
                pBaseJ++; pMatchRowJmD++;
                *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = cost;
            }
        }
    }
    // last 3 lines are empty
    for (int i = height - 3; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int d = 0; d < dispCountCompressed; d++) {
                *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = 255;
            }
        }
    }
}

void matchWTA_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
    const uint32 factorUniq = (uint32)(1024*uniqueness);
    const sint32 disp = maxDisp+1;
    
    // find best by WTA
    float32* pDestDisp = dispImg;
    for (sint32 i=0;i < height; i++) {
        for (sint32 j=0;j < width; j++) {
            // WTA on disparity values
            
            uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i,j,0);
            uint16* pCostBase = pCost;
            uint32 minCost = *pCost;
            uint32 secMinCost = minCost;
            int secBestDisp = 0;

            const uint32 end = disp-1;//MIN(disp-1,j);
            if (end == (uint32)disp-1) {
                uint32 bestDisp = 0;

                for (uint32 loop =0; loop < end;loop+= 8) {
                    // load costs
                    const __m128i costs = _mm_load_si128((__m128i*)pCost);
                    // get minimum for 8 values
                    const __m128i b = _mm_minpos_epu16(costs);
                    const int minValue = _mm_extract_epi16(b,0);

                   if ((uint32)minValue < minCost) {
                        minCost = (uint32)minValue;
                        bestDisp = _mm_extract_epi16(b,1)+loop;
                   }
                    pCost+=8;
                }

                // get value of second minimum
                pCost = pCostBase;
                pCost[bestDisp]=65535;

#ifdef USE_AVX2
                __m256i secMinVector = _mm256_set1_epi16(-1);
                const uint16* pCostEnd = pCost+disp;
                for (; pCost < pCostEnd;pCost+= 16) {
                    // load costs
                    __m256i costs = _mm256_load_si256((__m256i*)pCost);
                    // get minimum for 8 values
                    secMinVector = _mm256_min_epu16(secMinVector,costs);
                }
                secMinCost = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector,0)),0);
                uint32 secMinCost2 = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 1)), 0);
                if (secMinCost2 < secMinCost)
                    secMinCost = secMinCost2;
#else
                __m128i secMinVector = _mm_set1_epi16(-1);
                const uint16* pCostEnd = pCost+disp;
                for (; pCost < pCostEnd;pCost+= 8) {
                    // load costs
                    __m128i costs = _mm_load_si128((__m128i*)pCost);
                    // get minimum for 8 values
                    secMinVector = _mm_min_epu16(secMinVector,costs);
                }
                secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);
#endif
                pCostBase[bestDisp]=(uint16)minCost;
                
                // assign disparity
                if (1024*minCost <=  secMinCost*factorUniq) {
                    *pDestDisp = (float)bestDisp;

                } else {
                    bool check = false;
                    if (bestDisp < (uint32)maxDisp-1 && pCostBase[bestDisp+1] == secMinCost) {
                        check=true;
                    } 
                    if (bestDisp>0 && pCostBase[bestDisp-1] == secMinCost) {
                        check=true;
                    }
                    if (!check) {
                        *pDestDisp = -10;
                    } else {
                        *pDestDisp = (float)bestDisp;
                    }
                }
                
            } else {
                int bestDisp = 0;
                // for start
                for (uint32 k=1; k <= end; k++) {
                    pCost += 1;
                    const uint16 cost = *pCost;
                    if (cost < secMinCost) {
                        if (cost < minCost) {
                            secMinCost = minCost;
                            secBestDisp = bestDisp;
                            minCost = cost;
                            bestDisp = k;
                        } else  {
                            secMinCost = cost;
                            secBestDisp = k;
                        }
                    }
                }
                // assign disparity
                if (1024*minCost <=  secMinCost*factorUniq || abs(bestDisp - secBestDisp) < 2) {
                    *pDestDisp = (float)bestDisp;
                } else {
                     *pDestDisp = -10;
                }  
            }
            pDestDisp++;
        }
    }
}

FORCEINLINE __m128 rcp_nz_ss(__m128 input) {
    __m128 mask = _mm_cmpeq_ss(_mm_set1_ps(0.0), input);
    __m128 recip = _mm_rcp_ss(input);
    return _mm_andnot_ps(mask, recip);
}

FORCEINLINE void setSubpixelValue(float32* dest, uint32 bestDisp, const sint32& c0,const sint32& c1,const sint32& c2)
{
    __m128 denom = _mm_cvt_si2ss(_mm_setzero_ps(),c2 - c0);
    __m128 left = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c0);
    __m128 right = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c2);
    __m128 lowerMin = _mm_min_ss(left, right);
    __m128 d_offset = _mm_mul_ss(denom, rcp_nz_ss(_mm_mul_ss(_mm_set_ss(2.0f),lowerMin)));
    __m128 baseDisp = _mm_cvt_si2ss(_mm_setzero_ps(),bestDisp);
    __m128 result = _mm_add_ss(baseDisp, d_offset);
    _mm_store_ss(dest,result);
}

void matchWTAAndSubPixel_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
    const uint32 factorUniq = (uint32)(1024*uniqueness); 
    const sint32 disp = maxDisp+1;

    // find best by WTA
    float32* pDestDisp = dispImg;
    for (sint32 i=0;i < height; i++) {
        for (sint32 j=0;j < width; j++) {
            // WTA on disparity values

            uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i,j,0);
            uint16* pCostBase = pCost;
            uint32 minCost = *pCost;
            uint32 secMinCost = minCost;
            int secBestDisp = 0;

            const uint32 end = MIN(disp-1,j);
            if (end == (uint32)disp-1) {
                uint32 bestDisp = 0;

                for (uint32 loop =0; loop < end;loop+= 8) {
                    // load costs
                    const __m128i costs = _mm_load_si128((__m128i*)pCost);
                    // get minimum for 8 values
                    const __m128i b = _mm_minpos_epu16(costs);
                    const int minValue = _mm_extract_epi16(b,0);

                    if ((uint32)minValue < minCost) {
                        minCost = (uint32)minValue;
                        bestDisp = _mm_extract_epi16(b,1)+loop;
                    }
                    pCost+=8;
                }

                // get value of second minimum
                pCost = pCostBase;
                pCost[bestDisp]=65535;

#ifndef USE_AVX2
                __m128i secMinVector = _mm_set1_epi16(-1);
                const uint16* pCostEnd = pCost+disp;
                for (; pCost < pCostEnd;pCost+= 8) {
                    // load costs
                    __m128i costs = _mm_load_si128((__m128i*)pCost);
                    // get minimum for 8 values
                    secMinVector = _mm_min_epu16(secMinVector,costs);
                }
                secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);
                pCostBase[bestDisp] = (uint16)minCost;
#else
                __m256i secMinVector = _mm256_set1_epi16(-1);
                const uint16* pCostEnd = pCost + disp;
                for (; pCost < pCostEnd; pCost += 16) {
                    // load costs
                    __m256i costs = _mm256_load_si256((__m256i*)pCost);
                    // get minimum for 8 values
                    secMinVector = _mm256_min_epu16(secMinVector, costs);
                }
                secMinCost = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 0)), 0);
                uint32 secMinCost2 = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 1)), 0);
                if (secMinCost2 < secMinCost)
                    secMinCost = secMinCost2;
                pCostBase[bestDisp] = (uint16)minCost;
#endif

                // assign disparity
                if (1024*minCost <=  secMinCost*factorUniq) {
                    *pDestDisp = (float)bestDisp;
                } else {
                    bool check = false;
                    if (bestDisp < (uint32)maxDisp-1 && pCostBase[bestDisp+1] == secMinCost) {
                        check=true;
                    } 
                    if (bestDisp>0 && pCostBase[bestDisp-1] == secMinCost) {
                        check=true;
                    }
                    if (!check) {
                        *pDestDisp = -10;
                    } else {
                        if (0 < bestDisp && bestDisp < (uint32)maxDisp-1) {
                            setSubpixelValue(pDestDisp, bestDisp, pCostBase[bestDisp-1],minCost, pCostBase[bestDisp+1]);
                        } else {
                            *pDestDisp = (float)bestDisp;
                        }
                        
                    }
                }

            } else {
                int bestDisp = 0;
                // for start
                for (uint32 k=1; k <= end; k++) {
                    pCost += 1;
                    const uint16 cost = *pCost;
                    if (cost < secMinCost) {
                        if (cost < minCost) {
                            secMinCost = minCost;
                            secBestDisp = bestDisp;
                            minCost = cost;
                            bestDisp = k;
                        } else  {
                            secMinCost = cost;
                            secBestDisp = k;
                        }
                    }
                }
                // assign disparity
                if (1024*minCost <=  secMinCost*factorUniq || abs(bestDisp - secBestDisp) < 2) {
                    if (0 < bestDisp && bestDisp < maxDisp-1) {
                        setSubpixelValue(pDestDisp, bestDisp, pCostBase[bestDisp-1],minCost, pCostBase[bestDisp+1]);
                    } else {
                        *pDestDisp = (float)bestDisp;
                    }
                } else {
                    *pDestDisp = -10;
                }
            }
            pDestDisp++;
        }
    }
}

void matchWTARight_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
    const uint32 factorUniq = (uint32)(1024*uniqueness); 

    const uint32 disp = maxDisp+1;
    
    //_ASSERT(disp <= 256);
    // MSDN version

    // My version of this check
    //if(disp>256)
    //{
    //    printf("disparity value given greater than 256. Program exiting!\n");
    //    exit(EXIT_FAILURE);
    //} // end of check

    //ALIGN32 uint16 store[256+32]
    ALIGN32 uint16 store[disp+32];
    store[15] = UINT16_MAX-1;
    store[disp+16] = UINT16_MAX-1;

    // find best by WTA
    float32* pDestDisp = dispImg;
    for (uint32 i=0;i < (uint32)height; i++) {
        for (uint32 j=0;j < (uint32)width;j++) {
            // WTA on disparity values
            int bestDisp = 0;
            uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i,j,0);
            sint32 minCost = *pCost;
            sint32 secMinCost = minCost;
            int secBestDisp = 0;
            const uint32 maxCurrDisp = MIN(disp-1, width-1-j);

            if (maxCurrDisp == disp-1) {

                // transfer to linear storage, slightly unrolled
                for (uint32 k=0; k <= maxCurrDisp; k+=4) {
                    store[k+16]=*pCost;
                    store[k+16+1]=pCost[disp+1];
                    store[k+16+2]=pCost[2*disp+2];
                    store[k+16+3]=pCost[3*disp+3];
                    pCost += 4*disp+4;
                }
                // search in there
                uint16* pStore = &store[16];
                const uint16* pStoreEnd = pStore+disp;
                for (; pStore < pStoreEnd; pStore+=8) {
                    // load costs
                    const __m128i costs = _mm_load_si128((__m128i*)pStore);
                    // get minimum for 8 values
                    const __m128i b = _mm_minpos_epu16(costs);
                    const int minValue = _mm_extract_epi16(b,0);

                    if (minValue < minCost) {
                        minCost = minValue;
                        bestDisp = _mm_extract_epi16(b,1)+(int)(pStore-&store[16]);
                    }
                    
                }

                // get value of second minimum
                pStore = &store[16];
                store[16+bestDisp]=65535;
#ifndef USE_AVX2
                __m128i secMinVector = _mm_set1_epi16(-1);
                for (; pStore < pStoreEnd;pStore+= 8) {
                    // load costs
                    __m128i costs = _mm_load_si128((__m128i*)pStore);
                    // get minimum for 8 values
                    secMinVector = _mm_min_epu16(secMinVector,costs);
                }
                secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);
#else
                __m256i secMinVector = _mm256_set1_epi16(-1);
                for (; pStore < pStoreEnd; pStore += 16) {
                    // load costs
                    __m256i costs = _mm256_load_si256((__m256i*)pStore);
                    // get minimum for 8 values
                    secMinVector = _mm256_min_epu16(secMinVector, costs);
                }
                secMinCost = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 0)), 0);
                int secMinCost2 = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 1)), 0);
                if (secMinCost2 < secMinCost)
                    secMinCost = secMinCost2;
#endif

                // assign disparity
                if (1024U*minCost <=  secMinCost*factorUniq) {
                    *pDestDisp = (float)bestDisp;
                } else {
                    bool check = (store[16+bestDisp+1] == secMinCost);
                    check = check  | (store[16+bestDisp-1] == secMinCost);
                    if (!check) {
                        *pDestDisp = -10;
                    } else {
                        *pDestDisp = (float)bestDisp;
                    }
                }
                pDestDisp++;
            } 
            else {
                // border case handling
                for (uint32 k=1; k <= maxCurrDisp; k++) {
                    pCost += disp+1;
                    const sint32 cost = (sint32)*pCost;
                    if (cost < secMinCost) {
                        if (cost < minCost) {
                            secMinCost = minCost;
                            secBestDisp = bestDisp;
                            minCost = cost;
                            bestDisp = k;
                        } else {
                            secMinCost = cost;
                            secBestDisp = k;
                        }
                    }
                }
                // assign disparity
                if (1024U*minCost <= factorUniq*secMinCost|| abs(bestDisp - secBestDisp) < 2  ) {
                    *pDestDisp = (float)bestDisp;
                } else {
                    *pDestDisp = -10;
                }
                pDestDisp++;
            }
        }
    }
 }

void doLRCheck(float32* dispImg, float32* dispCheckImg, const sint32 width, const sint32 height, const sint32 lrThreshold)
{
    float* dispRow = dispImg;
    float* dispCheckRow = dispCheckImg;
    for (sint32 i=0;i < height;i++) {
        for (sint32 j=0;j < width;j++) {
            const float32 baseDisp = dispRow[j];
            if (baseDisp >= 0 && baseDisp <= j) {
                const float matchDisp = dispCheckRow[(int)(j-baseDisp)];

                sint32 diff = (sint32)(baseDisp - matchDisp);
                if (abs(diff) > lrThreshold) {
                    dispRow[j] = -10; // occluded or false match
                }
            } else {
                dispRow[j] = -10;
            }
        }
        dispRow += width;
        dispCheckRow += width;
    }    
}

void doRLCheck(float32* dispRightImg, float32* dispCheckImg, const sint32 width, const sint32 height, const sint32 lrThreshold)
{
    float* dispRow = dispRightImg;
    float* dispCheckRow = dispCheckImg;
    for (sint32 i=0;i < height;i++) {
        for (sint32 j=0;j < width;j++) {
            const float32 baseDisp = dispRow[j];
            if (baseDisp >= 0 && j+baseDisp <= width) {
                const float matchDisp = dispCheckRow[(int)(j+baseDisp)];

                sint32 diff = (sint32)(baseDisp - matchDisp);
                if (abs(diff) > lrThreshold) {
                    dispRow[j] = -10; // occluded or false match
                }
            } else {
                dispRow[j] = -10;
            }
        }
        dispRow += width;
        dispCheckRow += width;
    }    
}


/*  do a sub pixel refinement by a parabola fit to the winning pixel and its neighbors */
void subPixelRefine(float32* dispImg, uint16* dsiImg, const sint32 width, const sint32 height, const sint32 maxDisp, sint32 method)
{
    const sint32 disp_n = maxDisp+1;

    /* equiangular */
    if (method == 0) {

        for (sint32 y = 0; y < height; y++)
        {
            uint16*  cost = getDispAddr_xyd(dsiImg, width, disp_n, y, 1, 0);
            float* disp = (float*)dispImg+y*width;

            for (sint32 x = 1; x < width-1; x++, cost += disp_n)
            {
                if (disp[x] > 0.0) {

                    // Get minimum
                    int d_min = (int)disp[x];

                    // Compute the equations of the parabolic fit
                    uint16* costDmin = cost+d_min;
                    sint32 c0 = costDmin[-1], c1 = *costDmin, c2 = costDmin[1];

                    __m128 denom = _mm_cvt_si2ss(_mm_setzero_ps(),c2 - c0);
                    __m128 left = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c0);
                    __m128 right = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c2);
                    __m128 lowerMin = _mm_min_ss(left, right);
                    __m128 result = _mm_mul_ss(denom, rcp_nz_ss(_mm_mul_ss(_mm_set_ss(2.0f),lowerMin)));

                    __m128 baseDisp = _mm_cvt_si2ss(_mm_setzero_ps(),d_min);
                    result = _mm_add_ss(baseDisp, result);
                    _mm_store_ss(disp+x,result);

                } else {
                    disp[x] = -10;
                }
            }
        }
        /* 1: parabolic */
    } else if (method == 1){
        for (sint32 y = 0; y < height; y++)
        {
            uint16*  cost = getDispAddr_xyd(dsiImg, width, disp_n, y, 1, 0);
            float32* disp = dispImg+y*width;

            for (sint32 x = 1; x < width-1; x++, cost += disp_n)
            {
                if (disp[x] > 0.0) {

                    // Get minimum, but offset by 1 from ends
                    int d_min = (int)disp[x];

                    // Compute the equations of the parabolic fit
                    sint32 c0 = cost[d_min-1], c1 = cost[d_min], c2 = cost[d_min+1];
                    sint32 a = c0+c0 - 4 * c1 + c2+c2;
                    sint32 b =  (c0 - c2);

                    // Solve for minimum, which is a correction term to d_min
                    disp[x] = d_min + b /(float32) a;

                } else {
                    disp[x] = -10;
                }
            }
        }
    } else {
//        assert("subpixel interpolation method nonexisting");
    }
}

void uncompressDisparities_SSE(float32* dispImg, const sint32 width, const sint32 height, uint32 stepwidth)
{
    float32* disp = dispImg;
    const uint32 size = height*width;
    float32* dispEnd = disp + size;
    
    __m128 maskAll64 = _mm_set1_ps(64.f);

    if (stepwidth == 2) {
        while( disp < dispEnd)
        {
            __m128 valuesComp = _mm_load_ps(disp);

            __m128 result = _mm_add_ps(valuesComp, _mm_max_ps(_mm_sub_ps(valuesComp, maskAll64),_mm_setzero_ps()));
            _mm_store_ps(disp, result);
            disp+=4;
        }
    } else if (stepwidth == 4) {
        while( disp < dispEnd)
        {
            __m128 valuesComp = _mm_load_ps(disp);

            __m128 result = _mm_add_ps(valuesComp, _mm_mul_ps(_mm_max_ps(_mm_sub_ps(valuesComp, maskAll64),_mm_setzero_ps()), _mm_set1_ps(3.0f)));
            _mm_store_ps(disp, result);
            disp+=4;
        }
    }
}
