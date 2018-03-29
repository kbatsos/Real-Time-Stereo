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

__global__ void SADKernel(const float* left, const float* right, float* integral_vol,  const int rows, const int cols, const int integrrows , const int integrcols , const int ndisp ){

    extern __shared__ float row_slice[];

    int Col =  threadIdx.x;
    int uCol = XDIM_MAX_THREADS+threadIdx.x;
    int Row = blockIdx.x;

    if(Col <cols && Row<rows ){
        row_slice[Col] = left[Row*cols+Col];
        row_slice[cols+Col] = right[Row*cols+Col];
    }

    if(uCol<cols && Row<rows){
        row_slice[uCol] = left[Row*cols+uCol];
        row_slice[cols+uCol] = right[Row*cols+uCol];
    }

    __syncthreads();

    for(int d=0; d<ndisp; d++){

        if(Row < rows && Col < cols && Col-d >=0 ){
            integral_vol[d*integrrows*integrcols+Row*integrcols + Col] = abs(row_slice[Col] - row_slice[cols+Col-d] );
        }

        if(Row<rows && uCol<cols){
            integral_vol[d*integrrows*integrcols+Row*integrcols + uCol] = abs(row_slice[uCol] - row_slice[cols+uCol-d] );
        }

    }

}


__global__ void SAD(const float* integral_vol, float* slice, const int integrrows , const int integrcols , const int rows , const int cols, const int wsize, const int wc){
    
    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int d = blockIdx.z;

    float cost = 6375;
            

    if(Row < rows-wsize && Col < cols-wsize && Col >=d){
            cost = integral_vol[d*integrrows*integrcols+(Row+wsize)*integrcols + (Col+wsize)]
                                            - integral_vol[d*integrrows*integrcols+(Row)*integrcols + (Col+wsize)]
                                            - integral_vol[d*integrrows*integrcols+(Row+wsize)*integrcols + Col]
                                            + integral_vol[d*integrrows*integrcols+(Row)*integrcols + Col];
        
    }

    if(Row < rows && Col<cols)
        slice[d*rows*cols+(Row+wc)*cols+(Col+wc)] = cost;
}



void usage(void){
	std::cout	<< "SAD genmeric CUDA implementation" << std::endl;
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

int main(int argc, char* argv[]){

	string leftfile;
	string rightfile;
	string out=string("out");
	string out_t=string("disp");
	int wsize=9; 
	int ndisp=256;
	bool post=false;

	bool single=true;

	int argsassigned = 0;
	int required=0;

	postparams params;

	 //sgm params
    params.pi1=750;
    params.pi2=6000;
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
	else if( required < 4 ){
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

	int width = imgutil->getWidth();
    int height = imgutil->getHeight();

    int wdiv = ceil((float)width/32);

	cudaStream_t stream1;
    cudaStream_t stream2;
 
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    const int wc = wsize/2;
    float* cost_d;
    size_t bytes = height*width*ndisp*sizeof(float);
    cudaMalloc( (void**) &cost_d, bytes ); 

    float* post_cost_d;
    cudaMalloc( (void**) &post_cost_d, bytes ); 


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




    int size1 = height*ndisp;
    int size2 = width*ndisp;

    dim3 argGridSGM1((size1 - 1) / ndisp + 1,width);
    dim3 argGridSGM2((size2 - 1) / ndisp + 1,height);

    float * tmp_d;
	cudaMalloc(&tmp_d, width*ndisp*sizeof(float));
	cudaMemsetAsync(tmp_d,0 , width*ndisp*sizeof(float),0);

	float* left_cross;
	cudaMalloc(&left_cross, 4*height*width*sizeof(float));
	cudaMemsetAsync(left_cross,0 , 4*height*width*sizeof(float),0);

	float* right_cross;
	cudaMalloc(&right_cross, 4*height*width*sizeof(float));
	cudaMemsetAsync(right_cross,0 , 4*height*width*sizeof(float),0);


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
    cudaMemcpy( kernel_d, kernel, ks*ks*sizeof(float), cudaMemcpyHostToDevice);


    int vthreads = XDIM_MAX_THREADS;

    if(height < XDIM_Q_THREADS)
    	vthreads=XDIM_Q_THREADS;
    else if(height < XDIM_Q_THREADS)
    	vthreads=XDIM_H_THREADS;

    dim3 vintegralBlock(vthreads);	
    dim3 integralBlock(XDIM_MAX_THREADS); 

    dim3 integraldim1Griddisp(1,height,ndisp );
    dim3 vintegralGriddisp(1,width,ndisp );

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimCostSlice( ceil((float) width/ BLOCK_SIZE),ceil( (float)height/ BLOCK_SIZE),ndisp );

    dim3 swapBlock(BLOCK_D_SIZE,16,1);
    dim3 swapGrid(ceil((float)imgutil->getWidth()*imgutil->getHeight()/BLOCK_D_SIZE),ceil((float) ndisp/BLOCK_D_SIZE ));

    dim3 argBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 argGrid(ceil((float) imgutil->getWidth() / BLOCK_SIZE),ceil( (float)imgutil->getHeight()/ BLOCK_SIZE));

    dim3 SADBlock(XDIM_MAX_THREADS);
    dim3 SADGrid(imgutil->getHeight());

    int hreps = ceil((float) width/XDIM_MAX_THREADS );
    int vreps = ceil((float) height/vthreads );

    //###########################################################################################################################################//


	for(size_t i=0; i<limg.size(); i++){


    	imgutil->read_image(limg[i],imgl);
		imgutil->read_image(rimg[i],imgr);


	    cudaMemcpyAsync( imgl_d, imgl, width*height*sizeof(float), cudaMemcpyHostToDevice,stream1);
	    cudaMemcpyAsync( imgr_d, imgr, width*height*sizeof(float), cudaMemcpyHostToDevice,stream2);

	    
	    cudaMemsetAsync(cost_d,0 , height*width*ndisp*sizeof(float),stream1);
	    cudaMemsetAsync(post_cost_d,0 , width*height*ndisp*sizeof(float),stream2);

	    SADKernel<<<SADGrid, SADBlock,2*width*sizeof(float)>>>(imgl_d,imgr_d,post_cost_d,height , width,height , width,ndisp);


	    

	    for(int r=0; r<hreps; r++){
    		IntegralKernel<<<integraldim1Griddisp, integralBlock, 32*sizeof(float) >>>(post_cost_d, height, width,ndisp,r*(XDIM_MAX_THREADS-1) );
    	}

    	
    	
    	for(int r=0; r<vreps; r++){
    		VerticalIntegralKernel<<<vintegralGriddisp, vintegralBlock,32*sizeof(float) >>>(post_cost_d,height,width,1,r*(vthreads-1));
    	}

        SAD<<<dimCostSlice, dimBlock,0>>>(post_cost_d,cost_d,height,width,height,width ,wsize,wc);

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
    delete imgutil;
}