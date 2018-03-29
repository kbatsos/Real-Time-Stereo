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




__global__ void NCC(float* left, float* right, double* cost, const int rows, const int cols,const int ndisp){

    const int Col =blockIdx.x*blockDim.x + threadIdx.x-blockIdx.x*3;

    extern __shared__  __align__(sizeof(double)) unsigned char ncc_shared[];
    uint64 * Ar_sm = reinterpret_cast<uint64 *>(&ncc_shared[0]);
    double * Cr_sm = reinterpret_cast<double *>(&ncc_shared[(XDIM_H_THREADS+ndisp)*sizeof(double)]);

    float * l_im_sm = reinterpret_cast<float *>(&ncc_shared[  2*(XDIM_H_THREADS+ndisp)*sizeof(double)]);
    float * r_im_sm = reinterpret_cast<float *>(&ncc_shared[  2*(XDIM_H_THREADS+ndisp)*sizeof(double) + 3*(XDIM_H_THREADS)*sizeof(float) ]);

    int wh,ww;

    uint64 Al=0;
    uint64 temp1,temp2,p;
    double Cl;

    float l_im_0,l_im_1,l_im_2,l_im_3,l_im_4,l_im_5,l_im_6,l_im_7,l_im_8;

    int threaddispl = 0;
    if(blockIdx.x >0){
        threaddispl=ndisp;
    }


    if(blockIdx.x >0 && threadIdx.x < ndisp && Col-ndisp >=0 ){
        #pragma unroll
        for(wh=0; wh<3;wh++){
                r_im_sm[wh*(XDIM_H_THREADS+ndisp)+threadIdx.x] =right[(blockIdx.y+wh)*cols+Col-ndisp]; 
        } 
    }


    if(Col < cols){

        #pragma unroll
        for(wh=0; wh<3;wh++){
                r_im_sm[wh*(XDIM_H_THREADS+ndisp)+threaddispl+ threadIdx.x] =right[(blockIdx.y+wh)*cols+Col]; 
        }    
        
        #pragma unroll
        for(wh=0; wh<3;wh++){
                l_im_sm[wh*XDIM_H_THREADS+threadIdx.x] =left[(blockIdx.y+wh)*cols+Col]; 
        }    
        
    }



    __syncthreads();

    if(blockIdx.x >0 && threadIdx.x < ndisp && Col-ndisp >=0  ){

            temp1=0;
            temp2=0;

            #pragma unroll
            for(ww=0; ww<3;ww++){
                for (wh=0; wh<3;wh++){

                    p = r_im_sm[wh*(XDIM_H_THREADS+ndisp)+threadIdx.x+ww];
                    temp1 += p;
                    temp2 += p*p;

                }
            }

            Ar_sm[threadIdx.x]=temp1;

            Cr_sm[threadIdx.x] = 1/(sqrt(9*temp2 - (double)( temp1 )*( temp1) ));

    }

  __syncthreads();

    if(Col < cols-3 && threadIdx.x < blockDim.x-3){


            temp1 =0;
            temp2 =0;

            #pragma unroll
            for(ww=0; ww<3;ww++){
                for (wh=0; wh<3;wh++){
                
                    p =  r_im_sm[wh*(XDIM_H_THREADS+ndisp)+threaddispl+ threadIdx.x+ww];
                    temp1 += p;
                    temp2 += p*p;

                }
            }
            Ar_sm[threaddispl+ threadIdx.x]=temp1;
            Cr_sm[threaddispl+ threadIdx.x] = 1/(sqrt(9*temp2 - (double)( temp1 )*( temp1) ));

    }

      __syncthreads();




        if(Col < cols-3 && threadIdx.x < blockDim.x-3 ){

              l_im_0 = l_im_sm[threadIdx.x];
              l_im_1 = l_im_sm[threadIdx.x+1];
              l_im_2 = l_im_sm[threadIdx.x+2];
              l_im_3 = l_im_sm[XDIM_H_THREADS+threadIdx.x];
              l_im_4 = l_im_sm[XDIM_H_THREADS+threadIdx.x+1];
              l_im_5 = l_im_sm[XDIM_H_THREADS+threadIdx.x+2];
              l_im_6 = l_im_sm[2*XDIM_H_THREADS+threadIdx.x];
              l_im_7 = l_im_sm[2*XDIM_H_THREADS+threadIdx.x+1];
              l_im_8 = l_im_sm[2*XDIM_H_THREADS+threadIdx.x+2];
          


            Al = l_im_0+l_im_1+l_im_2+l_im_3+l_im_4+l_im_5 +l_im_6+l_im_7+l_im_8;

            Cl = 1/(sqrt(9*(l_im_0*l_im_0+
                            l_im_1*l_im_1+
                            l_im_2*l_im_2+
                            l_im_3*l_im_3+
                            l_im_4*l_im_4+
                            l_im_5*l_im_5+
                            l_im_6*l_im_6+
                            l_im_7*l_im_7+
                            l_im_8*l_im_8) 
                            - (double)( Al*Al )));


    }



    if(Col < cols-3 && threadIdx.x < blockDim.x-3){

            for(int d=0; d < ndisp; d++){
                double ncccost =2;
                if(Col-d>=0){

                    if( isfinite(Cl) &&  isfinite(Cr_sm[ threaddispl+ threadIdx.x-d])){    
                        double D =   l_im_0 * r_im_sm[threaddispl+ threadIdx.x-d] +
                                     l_im_1 * r_im_sm[threaddispl+ threadIdx.x-d+1]+
                                     l_im_2 * r_im_sm[threaddispl+ threadIdx.x-d+2]+
                                     l_im_3 * r_im_sm[XDIM_H_THREADS+ndisp+threaddispl+ threadIdx.x-d]+
                                     l_im_4 * r_im_sm[XDIM_H_THREADS+ndisp+threaddispl+ threadIdx.x-d+1]+
                                     l_im_5 * r_im_sm[XDIM_H_THREADS+ndisp+threaddispl+ threadIdx.x-d+2]+
                                     l_im_6 * r_im_sm[2*(XDIM_H_THREADS+ndisp)+threaddispl+ threadIdx.x-d]+
                                     l_im_7 * r_im_sm[2*(XDIM_H_THREADS+ndisp)+threaddispl+ threadIdx.x-d+1]+
                                     l_im_8 * r_im_sm[2*(XDIM_H_THREADS+ndisp)+threaddispl+ threadIdx.x-d+2];


                      
                        ncccost = 1- ((double)(9*D- Al * Ar_sm[threaddispl+ threadIdx.x-d] )*Cl*Cr_sm[ threaddispl+ threadIdx.x-d]);
                    }

                }

                cost[d*rows*cols+(blockIdx.y+1)*cols + (Col+1)]=ncccost;
            }
        }

}

void usage(void){
    std::cout   << "NCC fixed window CUDA implementation" << std::endl;
    std::cout   << "Arguments" << std::endl;
    std::cout   << "-l:\t\t Left image  | File containing names of the left images" << std::endl;
    std::cout   << "-r:\t\t Right image | File containing the names of the right images" << std::endl;
    std::cout   << "-ndisp:\t\t Number of Disparities" << std::endl;
    std::cout   << "-dopost:\t Default false. If set, activates sgm cost optimization" << std::endl;
    std::cout   << "-list:\t\t Default is single file. If set, left and right files should be lists of images." << std::endl;
    std::cout   << "-out:\t\t Output directory for disparity images." << std::endl;
    std::cout   << "-out_type:\t Output image type. Supports pgm|pfm|png|disp(uint16 png format)." << std::endl;
    std::cout   << "-postconf:\t Optional configuration file for post-processing." << std::endl;
    std::cout   << "-h:\t\t Prints this help" << std::endl;
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
    int wsize=3;

    postparams params;

     //sgm params
    params.pi1=1.32;
    params.pi2=24.25;
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

    double* cost_d;
    size_t bytes = height*width*ndisp*sizeof(double); 
    cudaMalloc( (void**) &cost_d, bytes ); 

    double* post_cost_d;
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


    uint64* l_integral_d;
    uint64* r_integral_d;
    uint64* l_integral_d_t;
    uint64* r_integral_d_t;


    cudaMalloc(&l_integral_d, height*width*sizeof(uint64));
    cudaMemsetAsync(l_integral_d, 0,height*width*sizeof(uint64),stream1);
    cudaMalloc(&r_integral_d, height*width*sizeof(uint64));
    cudaMemsetAsync(r_integral_d, 0,height*width*sizeof(uint64),stream2);

    cudaMalloc(&l_integral_d_t, height*width*sizeof(uint64));
    cudaMemsetAsync(l_integral_d_t, 0,height*width*sizeof(uint64),stream1);
    cudaMalloc(&r_integral_d_t, height*width*sizeof(uint64));
    cudaMemsetAsync(r_integral_d_t, 0,height*width*sizeof(uint64),stream2);

    unsigned long long int * l_sq_integral_d;
    unsigned long long int * r_sq_integral_d;

    cudaMalloc(&l_sq_integral_d, height*width*sizeof(unsigned long long int));
    cudaMemsetAsync(l_sq_integral_d, 0,height*width*sizeof(unsigned long long int),stream1);
    cudaMalloc(&r_sq_integral_d, height*width*sizeof(unsigned long long int));
    cudaMemsetAsync(r_sq_integral_d, 0,height*width*sizeof(unsigned long long int),stream2);


    uint64 * Al_d;
    uint64 * Ar_d;
    double* Cl_d;
    double* Cr_d;

    cudaMalloc(&Al_d, height*width*sizeof(uint64));
    cudaMalloc(&Ar_d, height*width*sizeof(uint64));
    cudaMalloc(&Cl_d, height*width*sizeof(double));
    cudaMalloc(&Cr_d, height*width*sizeof(double));

    cudaMemsetAsync(Al_d,0,height*width*sizeof(uint64));
    cudaMemsetAsync(Ar_d,0,height*width*sizeof(uint64));
    cudaMemsetAsync(Cl_d,0,height*width*sizeof(double));
    cudaMemsetAsync(Cr_d,0,height*width*sizeof(double));


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

   	dim3 dimBlockNCC(XDIM_H_THREADS);
    dim3 dimGridNCC(ceil((float) imgutil->getWidth() / XDIM_H_THREADS),imgutil->getHeight()-wsize);

    
    //#################################################################################################################


    for(size_t i=0; i<limg.size(); i++){

        imgutil->read_image(limg[i],imgl);
        imgutil->read_image(rimg[i],imgr);

        cudaMemsetAsync(cost_d,0 , height*width*ndisp*sizeof(double),stream1);

	    cudaMemcpyAsync( imgl_d, imgl, width*height*sizeof(float), cudaMemcpyHostToDevice,stream1);
	    cudaMemcpyAsync( imgr_d, imgr, width*height*sizeof(float), cudaMemcpyHostToDevice,stream2);


        NCC<<<dimGridNCC, dimBlockNCC,(2*(XDIM_H_THREADS+ndisp )*sizeof(double) + 3*(XDIM_H_THREADS)*sizeof(float)+ 3*(XDIM_H_THREADS+ndisp)*sizeof(float) )>>>( 
                        imgl_d, imgr_d,cost_d,height,width,ndisp);
		  

	    if(post){

	    	swap_axis<<< swapGrid, swapBlock >>>( cost_d, post_cost_d,height,width,ndisp );

	    	cudaMemset(cost_d,0 , height*width*ndisp*sizeof(double));

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
    cudaFree(l_integral_d);
    cudaFree(r_integral_d);
    cudaFree(l_sq_integral_d);
    cudaFree(r_sq_integral_d);
    cudaFree(Al_d);
    cudaFree(Ar_d);
    cudaFree(Cl_d);
    cudaFree(Cr_d);
    delete imgutil;

}