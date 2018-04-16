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

#define USE_AVX2

using namespace std;
using namespace std::chrono;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef unsigned int uint;
typedef unsigned long long ull;
#include "../libs/post_helper.cpp"

void ncc(uint8* leftp, uint8 *rightp, double* cost, int* shape, int ndisp, int wsize){

  
    const int wc = wsize/2;
    const int sqwin = wsize*wsize;
    const int intgrrows = shape[0]+1;
    const int intgrcols = shape[1] +1;

    unsigned int * lintegral = (unsigned int *)calloc(intgrrows*intgrcols,sizeof(unsigned int));
    unsigned int * rintegral = (unsigned int *)calloc(intgrrows*intgrcols,sizeof(unsigned int));
    unsigned long long * sqlintegral = (unsigned long long *)calloc(intgrrows*intgrcols,sizeof(unsigned long long));
    unsigned long long * sqrintegral = (unsigned long long *)calloc(intgrrows*intgrcols,sizeof(unsigned long long));


#pragma omp parallel num_threads(12)
    {
		#pragma omp for
		for(int i=0; i<shape[0]; i++){
			const int row = i*shape[1];
			const int introw = (i+1)*intgrcols;
			for (int j=0; j<shape[1]; j++){
				lintegral[introw+j+1] = leftp[row+j];
				rintegral[introw+j+1] = rightp[row+j];
			}
		}
    }




 

#pragma omp parallel num_threads(12)
    {
		#pragma omp for
		for(int i=0; i<shape[0]; i++){
			const int row = i*shape[1];
			const int introw = (i+1)*intgrcols;
			for (int j=0; j<shape[1]; j++){
				sqlintegral[introw+j+1] = leftp[row+j]*leftp[row+j];
				sqrintegral[introw+j+1] =  rightp[row+j]* rightp[row+j];
			}
		}
    }	 

  

#pragma omp parallel num_threads(12)
    {
		for (int i=1; i< intgrrows; i++){
			const int row = i*intgrcols;
			const int prev_row = (i-1)*intgrcols;
			#pragma omp for
			for(int j=0; j<intgrcols;j++){
				lintegral[row+j] += lintegral[prev_row+j];
				rintegral[row+j] += rintegral[prev_row+j];
			}
		}
    }


#pragma omp parallel num_threads(12)
    {
		for (int i=1; i< intgrrows; i++){
			const int row = i*intgrcols;
			const int prev_row = (i-1)*intgrcols;
			#pragma omp for
			for(int j=0; j<intgrcols;j++){
				sqlintegral[row+j] += sqlintegral[prev_row+j];
				sqrintegral[row+j] += sqrintegral[prev_row+j];
			}
		}
    }    

 

		


#pragma omp parallel num_threads(12)
    {
		#pragma omp for
		for(int i=0; i<intgrrows; i++){
			const int row =  i*intgrcols;
			for(int j=1; j<intgrcols; j++){
				lintegral[row+j] += lintegral[row+j-1];
				rintegral[row+j] += rintegral[row+j-1];
				sqlintegral[row+j] += sqlintegral[row+j-1];
				sqrintegral[row+j] += sqrintegral[row+j-1];
			}
		}
    }

    


    uint64* Al = (uint64 *)calloc(shape[0]*shape[1],sizeof(uint64));
    uint64* Ar = (uint64 *)calloc(shape[0]*shape[1],sizeof(uint64));
 
    double* Cl = (double *)calloc(shape[0]*shape[1],sizeof(double));
    double* Cr = (double *)calloc(shape[0]*shape[1],sizeof(double));



#pragma omp parallel num_threads(12)
    {

    	
		#pragma omp for
		for (int i=0; i< shape[0]-wsize;i++){
			const int row = (i+wc)*shape[1];
			const int t_row = i*intgrcols;
			const int b_row = (i+wsize)*intgrcols;
			for(int j=0; j< shape[1]-wsize; j++){
				const int col = j+wc;
				Al[row+col] = lintegral[b_row + j+wsize]+ lintegral[t_row + j]  - lintegral[b_row + j] - lintegral[t_row + j+wsize];
				Ar[row+col] = rintegral[b_row + j+wsize]+ rintegral[t_row + j] 	- rintegral[b_row + j] - rintegral[t_row + j+wsize];

			}
		}
    }

#pragma omp parallel num_threads(12)
    {
   	
		#pragma omp for
		for (int i=0; i< shape[0]-wsize;i++){
			const int row = (i+wc)*shape[1];
			const int t_row = i*intgrcols;
			const int b_row = (i+wsize)*intgrcols;
			int j=0;
			#ifdef USE_AVX2
			for(; j< shape[1]-wsize-4; j+=4){
				const int col = j+wc;
  
				__m256i ymm1 = _mm256_set_epi64x( sqlintegral[b_row + j+wsize], sqlintegral[b_row + j+wsize+1],sqlintegral[b_row + j+wsize+2],sqlintegral[b_row + j+wsize+3]  );
				__m256i ymm2 = _mm256_set_epi64x( sqlintegral[b_row + j], sqlintegral[b_row + j+1], sqlintegral[b_row + j+2],sqlintegral[b_row + j+3] );
				__m256i ymm3 = _mm256_sub_epi64(ymm1,ymm2);
				ymm2 = _mm256_set_epi64x( sqlintegral[t_row + j], sqlintegral[t_row + j+1], sqlintegral[t_row + j+2],sqlintegral[t_row + j+3] );
				ymm3 = _mm256_add_epi64(ymm3,ymm2);
				ymm1 = _mm256_set_epi64x( sqlintegral[t_row + j+wsize], sqlintegral[t_row + j+wsize+1], sqlintegral[t_row + j+wsize+2],sqlintegral[t_row + j+wsize+3] );
				ymm3 = _mm256_sub_epi64(ymm3,ymm1);

				__m256i ymm4 = _mm256_set_epi64x( sqrintegral[b_row + j+wsize], sqrintegral[b_row + j+wsize+1],sqrintegral[b_row + j+wsize+2],sqrintegral[b_row + j+wsize+3]  );
				__m256i ymm5 = _mm256_set_epi64x( sqrintegral[b_row + j],sqrintegral[b_row + j+1], sqrintegral[b_row + j+2],sqrintegral[b_row + j+3] );
				__m256i ymm6 = _mm256_sub_epi64(ymm4,ymm5);
				ymm5 = _mm256_set_epi64x( sqrintegral[t_row + j], sqrintegral[t_row + j+1], sqrintegral[t_row + j+2],sqrintegral[t_row + j+3] );
				ymm6 = _mm256_add_epi64(ymm6,ymm5);
				ymm4 = _mm256_set_epi64x( sqrintegral[t_row + j+wsize], sqrintegral[t_row + j+wsize+1], sqrintegral[t_row + j+wsize+2],sqrintegral[t_row + j+wsize+3] );
				ymm6 = _mm256_sub_epi64(ymm6,ymm4);
				 							

				ymm1 = _mm256_set_epi64x( Al[row+col], Al[row+col+1],Al[row+col+2],Al[row+col+3]  );
				ymm2 = _mm256_mul_epi32( ymm1,ymm1 );

				__m256d ymm1d = _mm256_set_pd( (double)_mm256_extract_epi64(ymm3,0),(double)_mm256_extract_epi64(ymm3,1), (double)_mm256_extract_epi64(ymm3,2),(double)_mm256_extract_epi64(ymm3,3) );
				__m256d ymm2d = _mm256_set1_pd((double) sqwin );
				__m256d ymm3d = _mm256_mul_pd (ymm1d, ymm2d);

				ymm1d = _mm256_set_pd( (double)_mm256_extract_epi64(ymm2,0),(double)_mm256_extract_epi64(ymm2,1), (double)_mm256_extract_epi64(ymm2,2),(double)_mm256_extract_epi64(ymm2,3) );
				ymm3d = _mm256_sub_pd (ymm3d, ymm1d);
				ymm3d = _mm256_sqrt_pd (ymm3d);
				ymm2d = _mm256_set1_pd((double) 1 );
				ymm3d = _mm256_div_pd(ymm2d,ymm3d);

				_mm256_storeu_pd(&Cl[row+col],ymm3d);

				if(!std::isfinite(Cl[ row+col]))
					Cl[ row+col ] = 0;
				if(!std::isfinite(Cl[ row+col+1]))
					Cl[ row+col+1 ] = 0;				
				if(!std::isfinite(Cl[ row+col+2]))
					Cl[ row+col+2 ] = 0;				
				if(!std::isfinite(Cl[ row+col+3]))
					Cl[ row+col+3 ] = 0;		


				ymm1 = _mm256_set_epi64x( Ar[row+col], Ar[row+col+1],Ar[row+col+2],Ar[row+col+3]  );
				ymm2 = _mm256_mul_epi32( ymm1,ymm1 );

				ymm1d = _mm256_set_pd( (double)_mm256_extract_epi64(ymm6,0),(double)_mm256_extract_epi64(ymm6,1), (double)_mm256_extract_epi64(ymm6,2),(double)_mm256_extract_epi64(ymm6,3) );
				ymm2d = _mm256_set1_pd((double) sqwin );
				ymm3d = _mm256_mul_pd (ymm1d, ymm2d);

				ymm1d = _mm256_set_pd( (double)_mm256_extract_epi64(ymm2,0),(double)_mm256_extract_epi64(ymm2,1), (double)_mm256_extract_epi64(ymm2,2),(double)_mm256_extract_epi64(ymm2,3) );
				ymm3d = _mm256_sub_pd (ymm3d, ymm1d);
				ymm3d = _mm256_sqrt_pd (ymm3d);
				ymm2d = _mm256_set1_pd((double) 1 );
				ymm3d = _mm256_div_pd(ymm2d,ymm3d);

				_mm256_storeu_pd(&Cr[row+col],ymm3d);							

				if(!std::isfinite(Cr[ row+col]))
					Cr[ row+col ] = 0;
				if(!std::isfinite(Cr[ row+col+1]))
					Cr[ row+col+1 ] = 0;				
				if(!std::isfinite(Cr[ row+col+2]))
					Cr[ row+col+2 ] = 0;				
				if(!std::isfinite(Cr[ row+col+3]))
					Cr[ row+col+3 ] = 0;							
							


			}
			#endif

			for(; j< shape[1]-wsize; j++){
				const int col = j+wc;

				unsigned long long Bl = sqlintegral[b_row + j+wsize] + sqlintegral[t_row + j] - sqlintegral[b_row + j] - sqlintegral[t_row + j+wsize];
				unsigned long long Br = sqrintegral[b_row + j+wsize] + sqrintegral[t_row + j] - sqrintegral[b_row + j] - sqrintegral[t_row + j+wsize];

				Cl[ row+col ] = 1/(sqrt(sqwin*Bl - (double)( Al[row+col] )*( Al[row+col] ) ));
				if(!std::isfinite(Cl[ row+col]))
					Cl[ row+col ] = 0;

				Cr[ row+col ] = 1/(sqrt(sqwin*Br - (double)( Ar[row+col] )*( Ar[row+col]) ));
				if(!std::isfinite(Cr[ row+col]))
					Cr[ row+col ] = 0;				


			}			
		}
    }   
       


      	
#pragma omp parallel num_threads(12)
    {

		double * dslice = (double*)calloc(intgrrows*intgrcols,sizeof(double));
		#pragma omp for
		for (int d=0; d<ndisp; d++ ){

			const int d_row = d*shape[0]*shape[1];
			std::fill_n(dslice,intgrrows*intgrcols,0);
			for(int i=0; i<shape[0]; i++){
				const int row = i*shape[1];
				const int intgrrow = (i+1)*intgrcols;
				for(int j=d; j<shape[1]; j++){
					dslice[intgrrow + j+1] = leftp[row+j]*rightp[row+j-d];
				}
			}

			for(int i=1; i<intgrrows; i++ ){
				const int row = i*intgrcols;
				const int prev_row = (i-1)*intgrcols;
				for(int j=0; j<intgrcols; j++){
					dslice[row + j] += dslice[prev_row + j];
				}

			}



 
		int iu=0;
		for( ; iu<intgrrows-8; iu+=8 ){
			const int rowind = iu*intgrcols;
			const int rowind1 = (iu+1)*intgrcols;
			const int rowind2 = (iu+2)*intgrcols;
			const int rowind3 = (iu+3)*intgrcols;
			const int rowind4 = (iu+4)*intgrcols;
			const int rowind5 = (iu+5)*intgrcols;
			const int rowind6 = (iu+6)*intgrcols;
			const int rowind7 = (iu+7)*intgrcols;			
			for(int j=d+1; j<intgrcols; j++){



				double s0, s1;
				s0 = dslice[rowind+j-1];
				s1 = dslice[rowind+j];
				dslice[rowind+j] = s1+s0;
				
				s0 = dslice[rowind1+j-1];
				s1 = dslice[rowind1+j];
				dslice[rowind1+j] = s1+s0;

				s0 = dslice[rowind2+j-1];
				s1 = dslice[rowind2+j];
				dslice[rowind2+j] = s1+s0;

				s0 = dslice[rowind3+j-1];
				s1 = dslice[rowind3+j];
				dslice[rowind3+j] = s1+s0;


				s0 = dslice[rowind4+j-1];
				s1 = dslice[rowind4+j];
				dslice[rowind4+j] = s1+s0;				

				s0 = dslice[rowind5+j-1];
				s1 = dslice[rowind5+j];
				dslice[rowind5+j] = s1+s0;

				s0 = dslice[rowind6+j-1];
				s1 = dslice[rowind6+j];
				dslice[rowind6+j] = s1+s0;			

				s0 = dslice[rowind7+j-1];
				s1 = dslice[rowind7+j];
				dslice[rowind7+j] = s1+s0;					

	
			}
		}

		for( ; iu<intgrrows; iu++){
			const int rowind = iu*intgrcols;
			for(int j=d+1; j<intgrcols; j++){
				dslice[rowind+j] += dslice[rowind+j-1];
			}
		}


			for (int i=0; i< shape[0]-wsize; i++){
				const int row = (i+wc)*shape[1];
				const int t_row = i*intgrcols;
				const int b_row = (i+wsize)*intgrcols;
				int j=d;
				#ifdef USE_AVX2 
				for(; j< shape[1]-wsize-4; j+=4){
					const int col = (j+wc);

					__m256d ymm1 = _mm256_loadu_pd (&dslice[b_row + j+wsize ]);
					__m256d ymm2 = _mm256_loadu_pd (&dslice[b_row +j ]);
					__m256d ymm3 = _mm256_sub_pd (ymm1, ymm2);
					ymm2 = _mm256_loadu_pd (&dslice[t_row+j]);
					ymm3 = _mm256_add_pd (ymm3, ymm2);
					ymm1 = _mm256_loadu_pd (&dslice[t_row +j+wsize ]);
					ymm3 = _mm256_sub_pd (ymm3, ymm1);
					ymm1 = _mm256_set1_pd((double) sqwin );
					ymm3 = _mm256_mul_pd (ymm3, ymm1);

					

					__m256i ymm4 = _mm256_set_epi64x( Al[row+col], Al[row+col+1], Al[row+col+2],Al[row+col+3] );
					__m256i ymm5 = _mm256_set_epi64x( Ar[row+(j-d+wc)], Ar[row+(j-d+wc)+1], Ar[row+(j-d+wc)+2],Ar[row+(j-d+wc)+3] );

					__m256i ymm6 = _mm256_mul_epi32( ymm4,ymm5 );

					ymm1 = _mm256_set_pd( (double)_mm256_extract_epi64(ymm6,0),(double)_mm256_extract_epi64(ymm6,1), (double)_mm256_extract_epi64(ymm6,2),(double)_mm256_extract_epi64(ymm6,3) );

					ymm3 = _mm256_sub_pd(ymm3,ymm1);

					

					ymm1 = _mm256_loadu_pd (&Cl[ row+col ]);
					ymm2 = _mm256_loadu_pd (&Cr[ row+(j-d+wc) ]);
					ymm2 = _mm256_mul_pd(ymm1,ymm2);
					ymm3 = _mm256_mul_pd(ymm3,ymm2);
					ymm1 = _mm256_set1_pd((double) -1 );
					ymm3 = _mm256_mul_pd(ymm3,ymm1);


					_mm256_storeu_pd(&cost[d_row + row+col],ymm3);

				}
				#endif

				for(; j< shape[1]-wsize; j++){
					const int col = (j+wc);
					const double lD = dslice[b_row + j+wsize ] + dslice[t_row+j]
								 	- dslice[b_row +j ] - dslice[t_row +j+wsize ];
								
			        	
			        cost[d_row + row+col] = -(sqwin*lD- Al[row+col] * Ar[row+(j-d+wc)]) *Cl[ row+col ]*Cr[ row+(j-d+wc) ];

				}				


			}

			

		}

		delete [] dslice;
    }
 
   

    delete [] lintegral;
    delete [] rintegral;
    delete [] sqlintegral;
    delete [] sqrintegral;

    delete [] Al;
    delete [] Ar;
    delete [] Cl;
    delete [] Cr;


}


void usage(void){
	std::cout	<< "NCC CPU generic implementation" << std::endl;
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


	string leftfile;
	string rightfile;
	string out=string(".");
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


	float scale = 32;

	std::vector<string> limg;
	std::vector<string> rimg;

	if (single){
		limg.push_back(leftfile);
		rimg.push_back(rightfile);

	}else{
		limg = getImages(leftfile);
		rimg = getImages(rightfile);
	}


	if(post){
		if( params.lrCheck ){
			if(ndisp%16 !=0){
				ndisp=ndisp+(16 - ndisp%16);
			}
		}else{
			if(ndisp%8!=0)
				ndisp=ndisp+(8 - ndisp%8);
		}
	}

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

	double* cost=(double*)calloc(width*imgutil->getHeight()*ndisp,sizeof(double));
	
	for(uint i=0; i< limg.size();i++){

		imgutil->read_image(limg[i],imgl);
		imgutil->read_image(rimg[i],imgr);


		if(post){
				imgl = paddwidth( imgl, imgutil->getHeight(), imgutil->getWidth() );
				imgr = paddwidth( imgr, imgutil->getHeight(), imgutil->getWidth() );
		}	
		
		std::fill_n(cost,shape[0]*shape[1]*ndisp, wsize*wsize-1);
		auto begin = std::chrono::high_resolution_clock::now();
		ncc(imgl, imgr, cost, shape, ndisp, wsize);



	 	if(post){
	 		doPost( cost, shape ,imgl,out + string("/") +limg[i].substr(limg[i].find_last_of("/")+1),out_t,scale,1,numStrips,params);
 		
		}else{
			argmin( cost,  disp, shape );
			imgutil->write_image(out + string("/") +limg[i].substr(limg[i].find_last_of("/")+1)  ,disp,out_t);
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
		std::cout << (float)duration/1000 <<  std::endl;

	}


	free(cost);
	free(imgl);
	free(imgr);
	free(disp);
	delete[] shape;
	delete imgutil;
	

	return 0;
}