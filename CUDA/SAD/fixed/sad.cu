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

__global__ void SAD(float* left, float* right, float* cost, int rows, int cols, int ndisp){

    const int shift = blockIdx.x*5;

    const int Row = blockIdx.y;
    const int Col =blockIdx.x*blockDim.x + threadIdx.x-shift;
    const int wc = 2;

    float l_im_0,l_im_1, l_im_2, l_im_3, l_im_4;
    __shared__ float abs_diff[XDIM_Q_THREADS];
    extern __shared__ float r_im_sm[];
    

    int threaddispl = 0;
    

    if(blockIdx.x >0){
        threaddispl=ndisp;
    }



    if( Col <cols ){


        l_im_0=left[(Row)*cols+Col ];
        l_im_1=left[(Row+1)*cols+Col ];
        l_im_2=left[(Row+2)*cols+Col ];
        l_im_3=left[(Row+3)*cols+Col ];
        l_im_4=left[(Row+4)*cols+Col];

        #pragma unroll
        for(int wh=0; wh<5;wh++){
                r_im_sm[wh*(XDIM_Q_THREADS+ndisp)+(threaddispl+ threadIdx.x)] =right[(Row+wh)*cols+Col]; 
        }        

    }

    float rp = ceil( (float)ndisp/blockDim.x  );

     for(int b=0; b<rp; b++){

	    if(blockIdx.x >0 && threadIdx.x < ndisp && (int)(Col-(ndisp-b*blockDim.x)) >=0 ){
	        #pragma unroll
	        for(int wh=0; wh<5;wh++){
	                r_im_sm[wh*(XDIM_Q_THREADS+ndisp)+(threadIdx.x+b*blockDim.x)] = right[(Row+wh)*cols+(Col -(ndisp-b*blockDim.x))]; 
	        } 
	    }
	}




        for(int d=0; d < ndisp; d++){
            float ab_dif=0;            
              
            if((int)(threaddispl+ threadIdx.x-d)>=0){   

	            ab_dif +=abs( l_im_0 - r_im_sm[threaddispl+ threadIdx.x-d] );
	            ab_dif +=abs( l_im_1 - r_im_sm[ (XDIM_Q_THREADS+ndisp)+ (threaddispl+ threadIdx.x-d)] );
	            ab_dif +=abs( l_im_2 - r_im_sm[2*(XDIM_Q_THREADS+ndisp)+(threaddispl+ threadIdx.x-d)] );
	            ab_dif +=abs( l_im_3 - r_im_sm[3*(XDIM_Q_THREADS+ndisp)+(threaddispl+ threadIdx.x-d)] );
	            ab_dif +=abs( l_im_4 - r_im_sm[4*(XDIM_Q_THREADS+ndisp)+(threaddispl+ threadIdx.x-d)] );

	            abs_diff[threadIdx.x]=ab_dif;  

        	}
            __syncthreads();
            if(Col < cols-5 ){
                    float sadcost =6375;
                        
                    if(Col-d>=0 && threadIdx.x <XDIM_Q_THREADS-5){

                        sadcost =ab_dif+abs_diff[threadIdx.x+1]+abs_diff[threadIdx.x+2]+abs_diff[threadIdx.x+3]+abs_diff[threadIdx.x+4]+abs_diff[threadIdx.x+5];
                    }
      
                    __syncthreads();   

                    if(threadIdx.x <XDIM_Q_THREADS-5){ 
                        
                        cost[d*rows*cols+(Row+wc)*cols + (Col+wc)]=sadcost;
                    }

            }

        }
    

}



void usage(void){
	std::cout	<< "SAD fixed window CUDA implementation" << std::endl;
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
	int wsize=5;

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

	int width = imgutil->getWidth();
    int height = imgutil->getHeight();

    int wdiv = ceil((float)width/32);

	cudaStream_t stream1;
    cudaStream_t stream2;
 
    
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);



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


    dim3 swapBlock(BLOCK_D_SIZE,16,1);
    dim3 swapGrid(ceil((float)imgutil->getWidth()*imgutil->getHeight()/BLOCK_D_SIZE),ceil((float) ndisp/BLOCK_D_SIZE ));

    dim3 argBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 argGrid(ceil((float) imgutil->getWidth() / BLOCK_SIZE),ceil( (float)imgutil->getHeight()/ BLOCK_SIZE));

    dim3 dimBlockSAD(XDIM_Q_THREADS); //
    dim3 dimGridSAD(ceil((float) imgutil->getWidth() / XDIM_Q_THREADS),imgutil->getHeight()-wsize);



    //###################################################################################################################################//

   for(size_t i=0; i<limg.size(); i++){


    	imgutil->read_image(limg[i],imgl);
		imgutil->read_image(rimg[i],imgr);


    	cudaMemsetAsync(cost_d,0 , height*width*ndisp*sizeof(float),stream1);
    	cudaMemsetAsync(post_cost_d,0 , width*height*ndisp*sizeof(float),stream2);


	    cudaMemcpyAsync( imgl_d, imgl, width*height*sizeof(float), cudaMemcpyHostToDevice,stream1);
	    cudaMemcpyAsync( imgr_d, imgr, width*height*sizeof(float), cudaMemcpyHostToDevice,stream2);

	    
        SAD<<<dimGridSAD, dimBlockSAD,5*(XDIM_Q_THREADS+ndisp)*sizeof(float)>>>( imgl_d, imgr_d,cost_d,height,width,ndisp);

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