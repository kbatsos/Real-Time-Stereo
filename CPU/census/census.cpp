#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <math.h> 
#include <immintrin.h>
#include <omp.h>
#include <sstream>
#include <chrono>
#include <limits>
#include <imgio.hpp>
#include "StereoBMHelper.h"
#include "FastFilters.h"
#include "StereoSGM.h"


using namespace std;
using namespace std::chrono;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef unsigned int uint;
typedef unsigned long long ull;

#include "../libs/post_helper.cpp"

void census(uint8* leftp, uint8 *rightp,uint *cost, int* shape, int ndisp, int wsize){


    int wc = wsize/2;
    int vecsize = wsize*wsize;
	if(vecsize%64 > 0)
		vecsize += 64-vecsize%64;
	int tchuncks = vecsize/64;

    uint64* censustl =  (uint64*)calloc( shape[0]*shape[1]*tchuncks,sizeof(uint64) ); 
    uint64* censustr =  (uint64*)calloc( shape[0]*shape[1]*tchuncks,sizeof(uint64) ); 
 
    #pragma omp parallel num_threads(12)
    {

    	#pragma omp for
		for (int i=0; i< shape[0]-wsize;i++){	
			const int row = (i+wc)*shape[1]*tchuncks;
			for(int j=0; j< shape[1]-wsize; j++){
				const int col = row + (j+wc)*tchuncks;
				for(int wh=0; wh<wsize; wh++){
					const uint8 lc = leftp[(i+wc)*shape[1]+(j+wc)];
					const uint8 rc = rightp[(i+wc)*shape[1]+(j+wc)];

					for(int ww=0; ww<wsize; ww++){
						int chunk =col+ ((wh*wsize+ww)/64);
						uint pos = ((wh*wsize+ww)&63);
						const int p = (i+wh)*shape[1]+j+ww;
						if( lc < leftp[p] ){
							censustl[ chunk ] ^= 1UL << pos ;
						}

						if( rc < rightp[p]  ){
							censustr[chunk ] ^= 1UL <<pos ;
						}

					}

				}


			}
		}


    }   
  

	#pragma omp parallel num_threads(12)
    {

		#pragma omp for
		for(int d=0; d < ndisp; d++){
			const int dind= d*shape[0]*shape[1];
			for (int i=0; i< shape[0]-wsize; i++){
				const int row = (i+wc)*shape[1];
				for(int j=d; j< shape[1]-wsize; j++){
					const int col = row*tchuncks + (j+wc)*tchuncks;
					const int cold = row*tchuncks + (j-d+wc)*tchuncks;
					uint sum =0;
					
					for(int ch=0; ch<tchuncks; ch++){
						uint64 r = censustl[col + ch ]^censustr[cold + ch ];
						sum +=_mm_popcnt_u64(r);

					}

					cost[dind + row + (j+wc)] = sum;

				}
			}
		}
    }    


    delete [] censustl;
    delete [] censustr;


}




void usage(void){
	std::cout	<< "Census CPU generic implementation" << std::endl;
	std::cout	<< "Arguments" << std::endl;
	std::cout	<< "-l:\t\t Left image  | File containing names of the left images" << std::endl;
	std::cout 	<< "-r:\t\t Right image | File containing the names of the right images" << std::endl;
	std::cout 	<< "-ndisp:\t\t Number of Disparities" << std::endl;
	std::cout 	<< "-wsize:\t\t Window size" << std::endl; 
	std::cout 	<< "-dopost:\t Default false. If set, activates sgm cost optimization" << std::endl;
	std::cout 	<< "-list:\t\t Default is single file. If set, left and right files should be lists of images." << std::endl;
	std::cout 	<< "-out:\t\t Output directory for disparity images." << std::endl;
	std::cout 	<< "-out_type:\t Output image type. Supports pgm|pfm|png|disp(uint16 png format)." << std::endl;
	std::cout 	<< "-postconf:\t Optional configuration file for post-processing." << std::endl;
	std::cout 	<< "-h:\t\t Prints this help" << std::endl;
}




int main(int argc,char* argv[]){

	high_resolution_clock::time_point t1;
	high_resolution_clock::time_point t2;

	string leftfile;
	string rightfile;
	string out=string("out");
	string out_t=string("disp");
	int ndisp=256;
	int wsize=9; 
	bool post=false;

	bool single=true;

	int argsassigned = 0;
	int required=0;

	int numStrips = 12;
    StereoSGMParams_t params;
    params.InvalidDispCost=63;
    params.lrCheck = true;
    params.MedianFilter = true;
    params.Paths = 8;
    params.subPixelRefine = 1;
    params.NoPasses = 2;
    params.rlCheck = false;


	for(int i=0; i<argc; i++){
		if( !strcmp(argv[i], "-l") ){
			leftfile = string(argv[++i]);
			argsassigned++;
			required++;
		}else if( !strcmp(argv[i],"-r") ){
			rightfile = string(argv[++i]);
			argsassigned++;
			required++;
		}else if( !strcmp(argv[i],"-ndisp") ){
			ndisp= atoi(argv[++i]);
			argsassigned++;
			required++;
		}else if( !strcmp(argv[i],"-wsize") ){
			wsize= atoi(argv[++i]);
			argsassigned++;
			required++;
		}else if( !strcmp(argv[i], "-dopost") ){
			post= true;
			argsassigned++;
		}else if(!strcmp(argv[i],"-list")){
			single=false;
			argsassigned++;
		}else if(!strcmp(argv[i],"-out")){
			out=string(argv[++i]);
			argsassigned++;
		}else if(!strcmp(argv[i],"-out_type")){
			out_t=string(argv[++i]);
			argsassigned++;
		}else if(!strcmp(argv[i],"-postconf")){
			parseConf(numStrips,params ,string(argv[++i]));
			argsassigned++;
		}else if(!strcmp(argv[i],"-h")){
			usage();
			return 0;
		}
	}

	if(argsassigned == 0){
		usage();
		return 0;
	}

	if(argsassigned ==1){
		leftfile = string("../../leftimg.txt");
		rightfile = string("../../rightimg.txt");
	}
	else if( required < 4 ){
		usage();
		return 0;
	}		


	float scale = (float)63/(wsize*wsize-1);

	std::vector<string> limg;
	std::vector<string> rimg;

	if (single){
		limg.push_back(leftfile);
		rimg.push_back(rightfile);

	}else{
		limg = getImages(leftfile);
		rimg = getImages(rightfile);
	}


	if(ndisp%8!=0)
		ndisp=ndisp+(8 - ndisp%8);

	imgio* imgutil = new imgio();
 
 	imgutil->read_image_meta(limg[0]);
	
	uint8* imgl= (uint8*)calloc(imgutil->getWidth()*imgutil->getHeight(),sizeof(uint8));
	uint8* imgr= (uint8*)calloc(imgutil->getWidth()*imgutil->getHeight(),sizeof(uint8));
	float* disp= (float*)calloc(imgutil->getWidth()*imgutil->getHeight(),sizeof(float));
	int* shape = new int[3];
	int width=imgutil->getWidth();
	shape[0]=imgutil->getHeight();shape[2]=ndisp;
	if(post){
 		width= imgutil->getWidth()+(16-imgutil->getWidth()%16);
 	}
 	shape[1]=width;

	uint* cost=(uint*)calloc(width*imgutil->getHeight()*ndisp,sizeof(uint));
	
	for(uint i=0; i< limg.size();i++){

		imgutil->read_image(limg[i],imgl);
		imgutil->read_image(rimg[i],imgr);


		if(post){
				imgl = paddwidth( imgl, imgutil->getHeight(), imgutil->getWidth() );
				imgr = paddwidth( imgr, imgutil->getHeight(), imgutil->getWidth() );
		}	
		
		std::fill_n(cost,shape[0]*shape[1]*ndisp, wsize*wsize-1);
		census(imgl, imgr, cost, shape, ndisp, wsize);


	 	if(post){
	 		doPost( cost, shape ,imgl,out + string("/") +limg[i].substr(limg[i].find_last_of("/")+1),out_t,scale,0,numStrips,params);
		}else{
			argmin( cost,  disp, shape );
			imgutil->write_image(out + string("/") +limg[i].substr(limg[i].find_last_of("/")+1)  ,disp,out_t);
		}

	}


	free(cost);
	free(imgl);
	free(imgr);
	free(disp);
	delete []shape;
	delete imgutil;
	

	return 0;
}