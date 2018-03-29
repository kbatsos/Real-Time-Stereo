#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <math.h> 
#include "imgio.hpp"
#include "post.cu"

using namespace std;
typedef uint8_t uint8;
typedef unsigned int uint32;
typedef unsigned long long int uint64;
#define STREAM_BLOCK 16
#define BLOCK_SIZE 32
#define BLOCK_D_SIZE 64
#define INTEGRAL_BLOCK_SIZE 8
#define XDIM_MAX_THREADS 1024
#define XDIM_H_THREADS 512
#define XDIM_Q_THREADS 256
#define SHARED_MEMORY 49152
#define INIT_BLOCK 8


__global__ void CensusTransformKernel(const float* image, uint64* census ,int rows, int cols ){




    __shared__ float cens_slice[7*XDIM_MAX_THREADS/2];
    const int Row = blockIdx.x;
    int Col = threadIdx.x;
    const int wr = 7/2; 
    const int wc = 9/2;
    const int steps = (ceil((double)cols/blockDim.x));


    for(int i=0; i<steps; i++){
        

        if(Col < cols){
            for(int wh=0; wh<7; wh++){
                cens_slice[threadIdx.x*7+wh] = image[(Row+wh)*cols + Col ];
            }
        }

        __syncthreads();



        if( Row < rows-7 && Col < cols-9 && threadIdx.x<blockDim.x-9 ){
                uint8 center = cens_slice[(threadIdx.x+wc)*7+wr];
                uint64 censtrans =0; 
                for(int ww=0; ww<9;ww++){ 
                    for ( int wh=0; wh<7;wh++){
                    	if( (center < cens_slice[(threadIdx.x+ww)*7+wh]) )
                        	censtrans ^= 1 << (wh*9+ww);
                    }
                }
                census[ (Row+wr)*cols + (Col+wc) ] = censtrans;



        }

        Col += blockDim.x - 9;  

        __syncthreads();                                    

    }

}


__global__ void CensusSADKernel(uint64* censusl, uint64* censusr, float* cost, int rows, int cols, int ndisp){

 	extern __shared__ uint64 censr_slice_sm[];

    uint64 censl_slice =0;
    const int Row = blockIdx.y;
    const int Col =blockIdx.x*blockDim.x + threadIdx.x; 
    const int wr = 7/2;
    const int wc = 9/2;
    int threaddispl = 0;
    if(blockIdx.x ==1){
    	threaddispl=ndisp;
    }


		if(Col >= cols && Col < (cols+ndisp) ){
			censr_slice_sm[Col-cols] = censusr[(Row+wr)*cols + (XDIM_MAX_THREADS + (Col-cols) -ndisp+wc) ];
		}

	    if(Col<cols){
	            censl_slice = censusl[(Row+wr)*cols + (Col+wc) ];
	            censr_slice_sm[threaddispl+ threadIdx.x ] = censusr[(Row+wr)*cols + (Col+wc) ];
	    
	            __syncthreads();

	    #pragma unroll
	    for (int d=0; d< ndisp; d++){

	        if(Col-d >=0 ){	            

	        	if(Col < cols-9){

	        		cost [ d*rows*cols+Row*cols+Col ]= (float)__popcll(censl_slice ^ censr_slice_sm[threaddispl+threadIdx.x-d ]);

	        	}
	           
	           

	         }
	    }

	}

	

}


void usage(void){
	std::cout	<< "Census fixed window CUDA implementation" << std::endl;
	std::cout	<< "Arguments" << std::endl;
	std::cout	<< "-l:\t\t Left image  | File containing names of the left images" << std::endl;
	std::cout 	<< "-r:\t\t Right image | File containing the names of the right images" << std::endl;
	std::cout 	<< "-ndisp:\t\t Number of Disparities" << std::endl;
	std::cout 	<< "-dopost:\t Default false. If set, activates sgm cost optimization" << std::endl;
	std::cout 	<< "-list:\t\t Default is single file. If set, left and right files should be lists of images." << std::endl;
	std::cout 	<< "-out:\t\t Output directory for disparity images." << std::endl;
	std::cout 	<< "-out_type:\t Output image type. Supports pgm|pfm|png|disp(uint16 png format)." << std::endl;
	std::cout 	<< "-postconf:\t Optional configuration file for post-processing." << std::endl;
	std::cout 	<< "-h:\t\t Prints this help" << std::endl;
}


int main(int argc, char* argv[]){

	string leftfile;
	string rightfile;
	string out=string("out");
	string out_t=string("disp");
	int ndisp=256;
	bool post=false;

	bool single=true;

	int argsassigned = 0;
	int required=0;

	postparams params;

	 //sgm params
    params.pi1=30;
    params.pi2=150;
    params.tau_so=1;
    params.alpha1=2;
    params.sgm_q1=3;
    params.sgm_q2=2;
    params.alpha2=6;
    params.sigma = 5.99;  
    params.kernel_size=5;  
    int direction =-1;
    



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
			parseConf(params ,string(argv[++i]));
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
	else if( required < 3 ){
		usage();
		return 0;
	}	
	std::vector<string> limg;
	std::vector<string> rimg;

	if (single){
		limg.push_back(leftfile);
		rimg.push_back(rightfile);

	}else{
		limg = getImages(leftfile);
		rimg = getImages(rightfile);
	}

	imgio* imgutil = new imgio();


	imgutil->read_image_meta(limg[0].c_str());

	



	//######################### Allocate memory on the device ###########################################//

	float* imgl;
    size_t ibytes = imgutil->getWidth()*imgutil->getHeight()*sizeof(float);
    cudaMallocHost( (void**) &imgl, ibytes );

    float* imgr;
    cudaMallocHost( (void**) &imgr, ibytes );

	cudaStream_t stream1;
    cudaStream_t stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
   
    float* cost_d;
    size_t bytes = imgutil->getWidth()*imgutil->getHeight()*ndisp*sizeof(float);
    cudaMalloc( (void**) &cost_d, bytes ); 

    float* post_cost_d;
    cudaMalloc( (void**) &post_cost_d, bytes ); 

    uint64 *census_l_d;
    cudaMalloc(&census_l_d, imgutil->getWidth()*imgutil->getHeight()*sizeof(uint64));
    cudaMemsetAsync(census_l_d, 0,imgutil->getWidth()*imgutil->getHeight()*sizeof(uint64),stream1);
    uint64 *census_r_d;
    cudaMalloc(&census_r_d, imgutil->getWidth()*imgutil->getHeight()*sizeof(uint64));
    cudaMemsetAsync(census_r_d, 0,imgutil->getWidth()*imgutil->getHeight()*sizeof(uint64),stream2);    

    float* disp_h;
    size_t dbytes = imgutil->getWidth()*imgutil->getHeight()*sizeof(float);
    cudaMallocHost( (void**) &disp_h, dbytes );
    float * disp_d;
    cudaMalloc(&disp_d, dbytes);
    float * disp_tmp;
    cudaMalloc(&disp_tmp, dbytes);

    float* imgl_d;
    cudaMalloc(&imgl_d, imgutil->getWidth()*imgutil->getHeight()*sizeof(float));
    float* imgr_d;
    cudaMalloc(&imgr_d, imgutil->getWidth()*imgutil->getHeight()*sizeof(float));


    dim3 dimBlockCens(XDIM_MAX_THREADS/2);
    dim3 dimGridCens(imgutil->getHeight()-7);

   	int threads = XDIM_MAX_THREADS;
	dim3 dimBlock(threads);
    dim3 dimGrid(ceil((float) imgutil->getWidth() /threads),imgutil->getHeight()-7);

    dim3 swapBlock(BLOCK_D_SIZE,16,1);
    dim3 swapGrid(ceil((float)imgutil->getWidth()*imgutil->getHeight()/BLOCK_D_SIZE),ceil((float)ndisp/BLOCK_D_SIZE));


    dim3 argBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 argGrid(ceil((float) imgutil->getWidth() / BLOCK_SIZE),ceil( (float)imgutil->getHeight()/ BLOCK_SIZE));

    int width = imgutil->getWidth();
    int height = imgutil->getHeight();

    int size1 = height*ndisp;
    int size2 = width*ndisp;

    float * tmp_d;
	cudaMalloc(&tmp_d, width*ndisp*sizeof(float));
	cudaMemsetAsync(tmp_d,0 , width*ndisp*sizeof(float),stream1);

	float* left_cross;
	cudaMalloc(&left_cross, 4*height*width*sizeof(float));
	cudaMemsetAsync(left_cross,0 , 4*height*width*sizeof(float),stream2);

	float* right_cross;
	cudaMalloc(&right_cross, 4*height*width*sizeof(float));
	cudaMemsetAsync(right_cross,0 , 4*height*width*sizeof(float),stream1);


	int kr = ceil(params.sigma*3);
	int ks = kr*2+1;
	float * kernel = (float*)calloc(ks*ks,sizeof(float));

	for (int i=0; i<ks; i++){
		for(int j=0; j<ks; j++){

			int y= (i-1)-kr;
			int x= (j-1)-kr;
			kernel[i*ks+j] = exp( -(x*x+y*y)/(2*params.sigma*params.sigma) );
		}
	}

	float *kernel_d;
	cudaMalloc(&kernel_d, ks*ks*sizeof(float));
    cudaMemcpyAsync( kernel_d, kernel, ks*ks*sizeof(float), cudaMemcpyHostToDevice,stream2);

    //#######################################################################################################################//

    for(size_t i=0; i<limg.size(); i++){

    	imgutil->read_image(limg[i],imgl);
		imgutil->read_image(rimg[i],imgr);

	    cudaMemcpyAsync( imgl_d, imgl, width*height*sizeof(float), cudaMemcpyHostToDevice,stream1);
	    cudaMemcpyAsync( imgr_d, imgr, width*height*sizeof(float), cudaMemcpyHostToDevice,stream2);

	    
	    cudaMemsetAsync(cost_d,80 , height*width*ndisp*sizeof(float),stream1);
	    

	    CensusTransformKernel<<<dimGridCens, dimBlockCens,0, stream1>>>(imgl_d,census_l_d,height, width);
		CensusTransformKernel<<<dimGridCens, dimBlockCens,0, stream2>>>(imgr_d,census_r_d,height, width);    
		
		CensusSADKernel<<<dimGrid, dimBlock,(threads+ndisp)*sizeof(uint64)>>>(census_l_d,census_r_d,cost_d,height, width,ndisp); 

		
		
	    if(post){

	    	swap_axis<<< swapGrid, swapBlock >>>( cost_d, post_cost_d,height,width,ndisp );

	    	cudaMemset(cost_d,0 , height*width*ndisp*sizeof(float));

			for (int step = 0; step < width; step++) {
				
				sgm_loop<0><<<(size1 - 1) / ndisp + 1, ndisp,2*ndisp*sizeof(float)>>>(
					imgl_d,
					imgr_d,
					
					post_cost_d,
					cost_d,

					tmp_d,
					params.pi1, params.pi2, params.tau_so, params.alpha1, params.sgm_q1, params.sgm_q2, direction,
					height,
					width,
					ndisp,
					step);
							
			}

			for (int step = 0; step < width; step++) {
				sgm_loop<1><<<(size1 - 1) / ndisp + 1, ndisp,2*ndisp*sizeof(float)>>>(
					imgl_d,
					imgr_d,
					
					post_cost_d,
					cost_d,

					tmp_d,
					params.pi1, params.pi2, params.tau_so, params.alpha1, params.sgm_q1, params.sgm_q2, direction,
					height,
					width,
					ndisp,
					step);
			}


			for (int step = 0; step < height; step++) {
				sgm_loop<2><<<(size2 - 1) / ndisp + 1, ndisp,2*ndisp*sizeof(float)>>>(
					imgl_d,
					imgr_d,
					
					post_cost_d,
					cost_d,

					tmp_d,
					params.pi1, params.pi2, params.tau_so, params.alpha1, params.sgm_q1, params.sgm_q2, direction,
					height,
					width,
					ndisp,
					step);
			}

			for (int step = 0; step < height; step++) {
				sgm_loop<3><<<(size2 - 1) / ndisp + 1, ndisp,2*ndisp*sizeof(float)>>>(
					imgl_d,
					imgr_d,
					
					post_cost_d,
					cost_d,

					tmp_d,
					params.pi1, params.pi2, params.tau_so, params.alpha1, params.sgm_q1, params.sgm_q2, direction,
					height,
					width,
					ndisp,
					step);
			}
 

		    argmin<<<argGrid, argBlock>>>( disp_d, cost_d, height, width,ndisp );

		    subpixel_enchancement<<<(height*width - 1) / TB + 1, TB>>>(
			disp_d,
			cost_d,
			disp_tmp,
			height*width,
			height*width,
			ndisp);



			median2d<<<(height*width - 1) / TB + 1, TB>>>(
			disp_tmp,
			disp_d,
			height*width,
			height,
			width,
			params.kernel_size / 2);

			mean2d<<<(height*width - 1) / TB + 1, TB>>>(
			disp_d,
			kernel_d,
			disp_tmp,
			height*width,
			ks / 2,
			height,
			width,
			params.alpha2);

		}else{

			argmin_d<<<argGrid, argBlock>>>( disp_tmp, cost_d, height, width,ndisp );
		}

	    cudaMemcpy( disp_h, disp_tmp, height*width*sizeof(float), cudaMemcpyDeviceToHost );

	    cudaError_t err = cudaGetLastError();
	    if (err != cudaSuccess) 
	        printf("Error: %s\n", cudaGetErrorString(err));
	    


	    imgutil->write_image(out + string("/") +limg[i].substr(limg[i].find_last_of("/")+1)  ,disp_h,out_t);

	}
	



	cudaFreeHost(imgl);
	cudaFreeHost(imgr);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	cudaFree(left_cross);
	cudaFree(right_cross);
   	cudaFree(tmp_d);
	cudaFreeHost(imgl);
	cudaFreeHost(imgr);
    cudaFreeHost(disp_h);
    cudaFree(disp_d);
    cudaFree(disp_tmp);
	cudaFree(imgl_d);
    cudaFree(imgr_d);
    cudaFree(cost_d);
    cudaFree(post_cost_d);
    cudaFree(census_l_d);
    cudaFree(census_r_d);
    delete imgutil;

	return 0;
}