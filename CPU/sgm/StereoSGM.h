// Copyright � Robert Spangenberg, 2014.
// See license.txt for more details

#pragma once

#include <vector>
#include <list>
#include <string.h>
#include <fstream>
#include <algorithm>

class StereoSGMParams_t {
public:
    uint16 P1; // +/-1 discontinuity penalty
    uint16 InvalidDispCost;  // init value for invalid disparities (half of max value seems ok)
    uint16 NoPasses; // one or two passes
    uint16 Paths; // 8, 0-4 gives 1D path, rest undefined
    float32 Uniqueness; // uniqueness ratio
    bool MedianFilter; // apply median filter
    bool lrCheck; // apply lr-check
    bool rlCheck; // apply rl-check (on right image)
    int lrThreshold; // threshold for lr-check
    int subPixelRefine; // sub pixel refine method

    // varP2 = - alpha * abs(I(x)-I(x-r))+gamma
    float32 Alpha; // variable P2 alpha
    uint16 Gamma; // variable P2 gamma
    uint16 P2min; // varP2 cannot get lower than P2min

    /* param set out of the paper from Banz 
    - noiseless (Cones): P1 = 11, P2min = 17, gamma = 35, alpha=0.5 8bit images 
    - Cones with noise: P1=20, P2min=24, gamma = 70, alpha=0.5
    */


    StereoSGMParams_t()
        : P1(5/*7*/)//5
        ,InvalidDispCost(128)
        ,NoPasses(2)
        ,Paths(8)
        ,Uniqueness(/*0.95f*/0.999f)
        ,MedianFilter(true)
        ,lrCheck(false)
        ,rlCheck(false)
        ,lrThreshold(1)
        ,subPixelRefine(-1)
        ,Alpha(0.25f/*0.25f*/)//0.25
        ,Gamma(24/*50*/)//24
        ,P2min(17)//12/*17*/
    {




    }
} ;

// template param is image pixel type (uint8 or uint16)
template <typename T>
class StereoSGM {
private:
    int m_width;
    int m_height;
    int m_maxDisp;
    StereoSGMParams_t m_params;
    uint16* m_S;

    float32* m_dispLeftImgUnfiltered;
    float32* m_dispRightImgUnfiltered;

    // SSE version, only maximum 8 paths supported
    template <int NPaths> void accumulateVariableParamsSSE(uint16* &dsi, T* img, uint16* &S);

public:
     // SGM
    StereoSGM(int i_width, int i_height, int i_maxDisp, StereoSGMParams_t i_params);
    ~StereoSGM();

    void process(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg);

    void processParallel(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg, int numThreads);

    // accumulation cube
    uint16* getS();
    // dimensions
    int getHeight();
    int getWidth();
    int getMaxDisp();
    // change params
    void setParams(const StereoSGMParams_t& i_params);

    void adaptMemory(int i_width, int i_height, int i_maxDisp);
};

template <typename T>
class StripedStereoSGM {
    int m_width;
    int m_height;
    int m_maxDisp;
    int m_numStrips;
    int m_border;

    std::vector<StereoSGM<T>* > m_sgmVector;
    std::vector<float32*> m_stripeDispImgVector;
    std::vector<float32*> m_stripeDispImgRightVector;

public:
    StripedStereoSGM(int i_width, int i_height, int i_maxDisp, int numStrips, const int border, StereoSGMParams_t i_params)
    {
        m_width = i_width;
        m_height = i_height;
        m_maxDisp = i_maxDisp;
        m_numStrips = numStrips;
        m_border = border;


        if (numStrips <= 1) {
            m_sgmVector.push_back(new StereoSGM<T>(m_width, m_height, m_maxDisp, i_params));
        }
        else {
            for (int n = 0; n < m_numStrips; n++)
            {
                sint32 startLine = n*(m_height / m_numStrips);
                sint32 endLine = (n + 1)*(m_height / m_numStrips) - 1;


                if (n == m_numStrips - 1) {
                    endLine = m_height - 1;
                }

                sint32 startLineWithBorder = MAX(startLine - m_border, 0);
                sint32 endLineWithBorder = MIN(endLine + m_border, m_height - 1);
                sint32 noLinesBorder = endLineWithBorder - startLineWithBorder + 1;

                m_stripeDispImgVector.push_back((float32*)_mm_malloc(noLinesBorder*m_width*sizeof(float32), 16));
                m_stripeDispImgRightVector.push_back((float32*)_mm_malloc(noLinesBorder*m_width*sizeof(float32), 16));

                m_sgmVector.push_back(new StereoSGM<T>(m_width, noLinesBorder, m_maxDisp, i_params));
            }
        }
    }
    ~StripedStereoSGM()
    {
        if (m_numStrips > 1) {
            for (int i = 0; i < m_numStrips; i++)
            {
                _mm_free(m_stripeDispImgVector[i]);
                _mm_free(m_stripeDispImgRightVector[i]);
                delete m_sgmVector[i];
            }
        }
        else {
            delete m_sgmVector[0];
        }
    }


    void writecostfile(uint16* dsi, char* name, int width, int height, int maxdisp,int cbin ){

        char filename[strlen(name)+4];

        if(cbin == 1){
        	sprintf(filename,"%s.bin",name);
        	FILE *fp;
        	fp = fopen(filename, "wb");
        	fwrite(dsi, sizeof(uint16), width*height*(maxdisp + 1), fp);
        	fclose(fp);
        }else{
        	std::ofstream ofs;
        	sprintf(filename,"%s.csv",name);
        	ofs.open(filename , std::ofstream::out);
            for(int i =0; i<height*width; i++){

                     for(int j=0; j<(maxdisp+1); j++){

                            ofs << dsi[i*(maxdisp+1) + j] << ",";


                    }
                 ofs << "\n";
            }
            ofs.close();
        }






     }

    void process(T* leftImg, float32* output, float32* dispImgRight, uint16* dsi, const int numThreads,int cbin)
    {

    	// uint16* costvol = (uint16*)_mm_malloc(m_width*m_height*(m_maxDisp + 1)*sizeof(uint16), 16);
        // no strips
        if (m_numStrips <= 1) {
            // normal processing (no strip)
            m_sgmVector[0]->process(dsi, leftImg, output, dispImgRight);

            return;
        }

        int NUM_THREADS = numThreads;
#pragma omp parallel num_threads(NUM_THREADS)
        {
#pragma omp for schedule(static, m_numStrips/NUM_THREADS)
            for (int n = 0; n < m_numStrips; n++){

                sint32 startLine = n*(m_height / m_numStrips);
                sint32 endLine = (n + 1)*(m_height / m_numStrips) - 1;

                if (n == m_numStrips - 1) {
                    endLine = m_height - 1;
                }

                sint32 startLineWithBorder = MAX(startLine - m_border, 0);

                int imgOffset = startLineWithBorder * m_width;
                int dsiOffset = startLineWithBorder * m_width * (m_maxDisp + 1);

                m_sgmVector[n]->process(dsi + dsiOffset, leftImg + imgOffset, m_stripeDispImgVector[n], m_stripeDispImgRightVector[n]);

                // copy back
                // int costOffset = (m_height / m_numStrips)*m_width * (m_maxDisp + 1);
                int upperBorder = startLine - startLineWithBorder;
                // memcpy(costvol + n*costOffset,m_sgmVector[n]->getS()+upperBorder*(m_width*(m_maxDisp + 1)),costOffset*sizeof(uint16));
                //memcpy(costvol + dsiOffset,m_sgmVector[n]->getS(),costOffset*sizeof(uint16));

                memcpy(output + startLine*m_width, m_stripeDispImgVector[n] + upperBorder*m_width, (endLine - startLine + 1)*m_width*sizeof(float32));
                memcpy(dispImgRight + startLine*m_width, m_stripeDispImgRightVector[n] + upperBorder*m_width, (endLine - startLine + 1)*m_width*sizeof(float32));
            }
        }

        	// for(int i=0; i<m_width*m_height*(m_maxDisp + 1);i++ ){
        	// 	dsi[i] = costvol[i];
        	// }
    }
};


#include "StereoSGM.hpp"
#include "StereoSGM_SSE.hpp"
