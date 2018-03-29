// Copyright � Robert Spangenberg, 2014.
// See license.txt for more details

#include "StereoCommon.h"
#include "StereoSGM.h"
#include <assert.h>

#include <algorithm>

template <typename T>
StereoSGM<T>::StereoSGM(int i_width, int i_height, int i_maxDisp, StereoSGMParams_t i_params)
    : m_width(i_width)
    , m_height(i_height)
    , m_maxDisp(i_maxDisp)
    , m_params(i_params)
{
    m_S = (uint16*) _mm_malloc(m_width*m_height*(i_maxDisp+1)*sizeof(uint16),16);

    m_dispLeftImgUnfiltered = (float*)_mm_malloc(m_width*m_height*sizeof(float), 16);
    m_dispRightImgUnfiltered = (float*)_mm_malloc(m_width*m_height*sizeof(float), 16);
}

template <typename T>
StereoSGM<T>::~StereoSGM()
{
    if (m_S != NULL)
        _mm_free(m_S); 

    if (m_dispLeftImgUnfiltered != NULL)
        _mm_free(m_dispLeftImgUnfiltered);
    if (m_dispRightImgUnfiltered != NULL)
        _mm_free(m_dispRightImgUnfiltered);
}

template <typename T>
void StereoSGM<T>::adaptMemory(int i_width, int i_height, int i_maxDisp)
{
    if (i_width*i_height*i_maxDisp > m_width * m_height*m_maxDisp) {
        if (m_S != NULL) {
            _mm_free(m_S);
        }
        m_width = i_width;
        m_height = i_height;
        m_maxDisp = i_maxDisp;
        m_S = (uint16*) _mm_malloc(m_width*m_height*(i_maxDisp+1)*sizeof(uint16),16);
    } else {
        m_width = i_width;
        m_height = i_height;
        m_maxDisp = i_maxDisp;
    }
}

template <typename T>
uint16* StereoSGM<T>::getS()
{
    return m_S;
}

template <typename T>
int StereoSGM<T>::getHeight()
{
    return m_height;
}

template <typename T>
int StereoSGM<T>::getWidth()
{
    return m_width;
}

template <typename T>
int StereoSGM<T>::getMaxDisp()
{
    return m_maxDisp;
}

template <typename T>
void StereoSGM<T>::setParams(const StereoSGMParams_t& i_params)
{
    m_params = i_params;
}


inline void swapPointers(uint16*& p1, uint16*& p2)
{
    uint16* temp = p1;
    p1 = p2;
    p2 = temp;
}

inline sint32 adaptP2(const float32& alpha, const uint16& I_p, const uint16& I_pr, const int& gamma, const int& P2min)
{
    sint32 result;
    result = (sint32)(-alpha * abs((sint32)I_p-(sint32)I_pr)+gamma);
    if (result < P2min)
        result = P2min;
    return result;
}

template <typename T>
void StereoSGM<T>::process(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg)
{


    if (m_params.Paths == 0) {
        accumulateVariableParamsSSE<0>(dsi, img, m_S);
    }
    else if (m_params.Paths == 1) {
        accumulateVariableParamsSSE<1>(dsi, img, m_S);
    }
    else if (m_params.Paths == 2) {
        accumulateVariableParamsSSE<2>(dsi, img, m_S);
    }
    else if (m_params.Paths == 3) {
        accumulateVariableParamsSSE<3>(dsi, img, m_S);
    }
    else if (m_params.Paths == 8) {

        accumulateVariableParamsSSE<8>(dsi, img, m_S);
        
    }


    // median filtering preparation
    float *dispLeftImgUnfiltered;
    float *dispRightImgUnfiltered;
    
    if (m_params.MedianFilter) {
        dispLeftImgUnfiltered = /*dispLeftImg;*/m_dispLeftImgUnfiltered; 
        dispRightImgUnfiltered = /*dispRightImg;*/m_dispRightImgUnfiltered;
    } else {
        dispLeftImgUnfiltered = dispLeftImg;
        dispRightImgUnfiltered = dispRightImg;
    }

    if (m_params.lrCheck) {
        matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
        matchWTARight_SSE(dispRightImgUnfiltered, m_S,m_width, m_height, m_maxDisp, m_params.Uniqueness);

        /* subpixel refine */
        if (m_params.subPixelRefine != -1) {
            subPixelRefine(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.subPixelRefine);
        }

        if (m_params.MedianFilter) {
            median3x3_SSE(dispLeftImgUnfiltered, dispLeftImg, m_width, m_height);
            median3x3_SSE(dispRightImgUnfiltered, dispRightImg, m_width, m_height);
        } 
        doLRCheck(dispLeftImg, dispRightImg, m_width, m_height, m_params.lrThreshold);

        if (m_params.rlCheck)
        {
            doRLCheck(dispRightImg, dispLeftImg, m_width, m_height, m_params.lrThreshold);
        }
        
    } else {
        // find disparities with minimum accumulated costs
        matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
        /* subpixel refine */
        if (m_params.subPixelRefine != -1) {
            subPixelRefine(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.subPixelRefine);
        }
        if (m_params.MedianFilter) {
            median3x3_SSE(dispLeftImgUnfiltered, dispLeftImg, m_width, m_height);
            median3x3_SSE(dispRightImgUnfiltered, dispRightImg, m_width, m_height);
        } 
    }
}

template <typename T>
void StereoSGM<T>::processParallel(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg, sint32 numThreads)
{
    
    if (m_params.Paths == 0) {
        accumulateVariableParamsSSE<0>(dsi, img, m_S);
    }
    else if (m_params.Paths == 1) {
        accumulateVariableParamsSSE<1>(dsi, img, m_S);
    }
    else if (m_params.Paths == 2) {
        accumulateVariableParamsSSE<2>(dsi, img, m_S);
    }
    else if (m_params.Paths == 3) {
        accumulateVariableParamsSSE<3>(dsi, img, m_S);
    }
    else if (m_params.Paths == 8) {
        accumulateVariableParamsSSE<8>(dsi, img, m_S);
    }

    // median filtering preparation
    float *dispLeftImgUnfiltered;
    float *dispRightImgUnfiltered;

    if (m_params.MedianFilter) {
        dispLeftImgUnfiltered = m_dispLeftImgUnfiltered;
        dispRightImgUnfiltered = m_dispRightImgUnfiltered;
    } else {
        dispLeftImgUnfiltered = dispLeftImg;
        dispRightImgUnfiltered = dispRightImg;
    }

    if (m_params.lrCheck) {
        // find disparities with minimum accumulated costs
        if (numThreads == 1) {
            matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
            matchWTARight_SSE(dispRightImgUnfiltered, m_S,m_width, m_height, m_maxDisp, m_params.Uniqueness);
        } else if (numThreads > 1) {
#pragma omp parallel num_threads(2)
            {
#pragma omp sections nowait
                {
#pragma omp section
                    {
                        if (m_params.subPixelRefine != -1) {
                            matchWTAAndSubPixel_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
                        } else {
                            matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
                        }
                    }
#pragma omp section
                    {
                        matchWTARight_SSE(dispRightImgUnfiltered, m_S,m_width, m_height, m_maxDisp, m_params.Uniqueness);
                    }
                }
            }
        }

        if (m_params.MedianFilter) {
            median3x3_SSE(dispLeftImgUnfiltered, dispLeftImg, m_width, m_height);
            median3x3_SSE(dispRightImgUnfiltered, dispRightImg, m_width, m_height);
        }
        doLRCheck(dispLeftImg, dispRightImg, m_width, m_height, m_params.lrThreshold);

        if (m_params.rlCheck)
        {
            doRLCheck(dispRightImg, dispLeftImg, m_width, m_height, m_params.lrThreshold);
        }
    } else {
        matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);

        /* subpixel refine */
        if (m_params.subPixelRefine != -1) {
            subPixelRefine(dispLeftImg, m_S, m_width, m_height, m_maxDisp, m_params.subPixelRefine);
        }
    }
}
