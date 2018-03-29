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
#define SHARED_MEMORY 49152
#define XDIM_H_THREADS 512
#define XDIM_Q_THREADS 256

__global__ void CensusTransformKernel(const float* image, uint64* census ,int rows, int cols, int wsize, int chunks ){



	const int shift = blockIdx.x*wsize;
    extern __shared__ float cens_slice[];
    int Row = blockIdx.y;
    int Col = blockIdx.x*blockDim.x+ threadIdx.x-shift; 
    int wc = wsize/2;
    float center;
    uint pos = 0;
    uint64 cens=0;
    int ch=0;

   	if(Col < cols-wsize && Row< rows-wsize)
	    center = image[(Row+wc)*cols + Col+wc ];



	if(Col < cols && Row< rows-wsize){

    	for(int i=0; i<wsize; i++){
    		cens_slice[threadIdx.x] = image[(Row+i)*cols + Col ];
	    		
	    	__syncthreads();

	    	if(threadIdx.x < blockDim.x-wsize && Col<cols-wsize){

		    	for(int ww=0; ww<wsize;ww++){ 
		    		if( center < cens_slice[threadIdx.x+ww])
		    			cens ^= 1UL << pos;
		    		pos++;

		    		if( (pos & 63) == 63 ){
		    			census[ ((Row+wc)*cols + (Col+wc))*chunks+ch ] = cens;
		    			ch++;
		    			cens=0;
		    			pos=0;
		    		}
		    	}
		    }
		    
		    __syncthreads();
		}


		if(threadIdx.x < blockDim.x-wsize && ch<chunks){
    		census[ ((Row+wc)*cols + (Col+wc))*chunks+ch] = cens;
    	}
    }
}

__global__ void CensusSADKernel(uint64* censusl, uint64* censusr, float* cost, int rows, int cols, int ndisp, int wsize,int maxcost,int chunks,int sm_offset ){

	extern __shared__ uint64 rc_sm[];

	const int Row = blockIdx.y;
    const int Col =blockIdx.x*blockDim.x + threadIdx.x;
    int wc = wsize/2;
    uint64* lbs = &rc_sm[sm_offset];

    int threaddispl = 0;
    if(blockIdx.x >0){
    	threaddispl=ndisp;
    }

    float rp = ceil( (float)ndisp/blockDim.x  );

    for(int b=0; b<rp; b++){
    
    	if(blockIdx.x > 0 && threadIdx.x < ndisp && (int)(Col -(ndisp-b*blockDim.x))>=0 ){
    	
    		for (int ch=0; ch< chunks; ch++){
         		rc_sm[(threadIdx.x+b*blockDim.x)*chunks+ch] = censusr[  ((Row+wc)*cols  +  (Col -(ndisp-b*blockDim.x)+wc))*chunks +ch  ];
    		}
    	}
    }



    __syncthreads();

    if(Row < rows-wsize && Col < cols-wsize){
    	const int index = ((Row+wc)*cols+ (Col+wc))*chunks;
    	for(int ch=0; ch< chunks; ch++){
    		
    		lbs[threadIdx.x*chunks+ch] = censusl[index+ch];
    		rc_sm[(threaddispl+ threadIdx.x )*chunks+ch] = censusr[index+ch ];
    	}

    	__syncthreads();


	    	for (int d=0; d< ndisp; d++){
	    		const int dindex = threaddispl+threadIdx.x-d;

	    		if(Col < cols-wsize && dindex >=0){

	    			float sum =0;
					
					for(int ch=0; ch<chunks; ch++){
						uint64 r = lbs[ threadIdx.x*chunks+ch ]^rc_sm[dindex*chunks + ch ];
						sum +=(float)__popcll(r);

					}

					cost[d*rows*cols+(Row+wc)*cols + (Col+wc)] = sum;
					
	    		}

	    	}

    }






}



__global__ void CensusTransformKernelgen(const float* image, uint64* census ,int rows, int cols, int wsize, int chunks ){



	const int shift = blockIdx.x*wsize;
    extern __shared__ float cens_slice[];
    int Row = blockIdx.y;
    int Col = blockIdx.x*blockDim.x+ threadIdx.x-shift; 
    int wc = wsize/2;
    float center;
    uint pos = 0;
    uint64 cens=0;
    int ch=0;

    

   	if(Col < cols-wsize && Row< rows-wsize){
	    center = image[(Row+wc)*cols + Col+wc ];
	   
   	}

   	__syncthreads();




	if(Col < cols && Row< rows-wsize){

    	for(int i=0; i<wsize; i++){
    		cens_slice[threadIdx.x] = image[(Row+i)*cols + Col ];
	    		
	    	__syncthreads();

	    	if(threadIdx.x < blockDim.x-wsize){

		    	for(int ww=0; ww<wsize;ww++){ 
		    		if( center < cens_slice[threadIdx.x+ww])
		    			cens ^= 1UL << pos;
		    		pos++;

		    		if( (pos & 63) == 63 ){
		    			census[ (ch*rows+ (Row+wc))*cols + (Col+wc) ] = cens;
		    			ch++;
		    			cens=0;
		    			pos=0;
		    		}
		    	}
		    }
		    
		    __syncthreads();
		}


		if(threadIdx.x < blockDim.x-wsize){			
    		census[ (ch*rows+ (Row+wc))*cols + (Col+wc) ] = cens;
    	}
    }

}

__global__ void CensusSADKernelgen(uint64* censusl, uint64* censusr, float* cost, int rows, int cols, int ndisp, int wsize,int maxcost,int chunks ){

	extern __shared__ uint64 rc_sm[];

	const int Row = blockIdx.y;
    const int Col =blockIdx.x*blockDim.x + threadIdx.x;
    int wc = wsize/2;
    uint64 lbs;

    int threaddispl = 0;
    if(blockIdx.x >0){
    	threaddispl=ndisp;
    }
   
    float rp = ceil( (float)ndisp/blockDim.x  );

	for(int ch=0; ch<chunks;ch++){

		for(int b=0; b<rp; b++){
	    	if(blockIdx.x > 0 && threadIdx.x < ndisp && (int)(Col -(ndisp-b*blockDim.x)) >=0){
	        	rc_sm[(threadIdx.x+b*blockDim.x)] = censusr[ (ch*rows + (Row+wc))*cols + (Col -(ndisp-b*blockDim.x)+wc)  ];
	    	}
	    }	


	    if(Row < rows-wsize && Col < cols-wsize){
			const int index = (ch*rows + (Row+wc))*cols + (Col+wc);
			lbs = censusl[index];
			rc_sm[threaddispl+ threadIdx.x] = censusr[index ];

		}   
		    	
		    __syncthreads();   
		    
		   	
		if(Row < rows-wsize && Col < cols-wsize){    
	    		for (int d=0; d< ndisp; d++){
	    			const int dind = threaddispl+threadIdx.x-d;
	    			if(dind >0){
	    				cost[(d*rows+(Row+wc))*cols + (Col+wc)] -= 64 - (float)__popcll(lbs^rc_sm[dind]);
	    			}
	    		}

	    }

	    	__syncthreads();  
    }
}

__global__ void inti_cost( float* cost, int rows, int cols, int ndisp,float maxcost ){


    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x; 

    if( Row < rows && Col < cols){
	    for(int i=0; i<ndisp; i++){
	    	cost[ i*rows*cols+Row*cols+Col ] = maxcost;
	    }

	}
}



void usage(void){
	std::cout	<< "Census genmeric CUDA implementation" << std::endl;
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
    params.pi1=30;
    params.pi2=500;
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


    float* cost_d;
    size_t bytes = imgutil->getWidth()*imgutil->getHeight()*ndisp*sizeof(float);
    cudaMalloc( (void**) &cost_d, bytes ); 

    float* post_cost_d;
    cudaMalloc( (void**) &post_cost_d, bytes ); 


    int vecsize = wsize*wsize;
	if(vecsize%64 > 0)
		vecsize += 64-(vecsize&63);
	int tchuncks = vecsize/64;   


	float maxcost = tchuncks*64;

    uint64 *census_l_d;
    cudaMalloc(&census_l_d, imgutil->getWidth()*imgutil->getHeight()*tchuncks*sizeof(uint64));
    cudaMemsetAsync(census_l_d, 0,imgutil->getWidth()*imgutil->getHeight()*tchuncks*sizeof(uint64),0);
    uint64 *census_r_d;
    cudaMalloc(&census_r_d, imgutil->getWidth()*imgutil->getHeight()*tchuncks*sizeof(uint64));
    cudaMemsetAsync(census_r_d, 0,imgutil->getWidth()*imgutil->getHeight()*tchuncks*sizeof(uint64),0);    

    float* disp_h;
    size_t dbytes = imgutil->getWidth()*imgutil->getHeight()*sizeof(float);
    cudaMallocHost( (void**) &disp_h, dbytes );

    float * disp_d;
    cudaMalloc(&disp_d, dbytes);
    cudaMemsetAsync(disp_d, 0,imgutil->getWidth()*imgutil->getHeight()*sizeof(float),0);
    float * disp_tmp;
    cudaMalloc(&disp_tmp, dbytes);
    cudaMemsetAsync(disp_tmp, 0,imgutil->getWidth()*imgutil->getHeight()*sizeof(float),0);

    float* imgl_d;
    cudaMalloc(&imgl_d, imgutil->getWidth()*imgutil->getHeight()*sizeof(float));
    float* imgr_d;
    cudaMalloc(&imgr_d, imgutil->getWidth()*imgutil->getHeight()*sizeof(float));

    dim3 swapBlock(BLOCK_D_SIZE,16,1);
    dim3 swapGrid(ceil((float)imgutil->getWidth()*imgutil->getHeight()/BLOCK_D_SIZE),ceil((float) ndisp/BLOCK_D_SIZE ));


    dim3 dimBlockCens(XDIM_MAX_THREADS);
    float blockx = (float)imgutil->getWidth() / XDIM_MAX_THREADS;
    dim3 dimGridCens(ceil((float) blockx) + (blockx*wsize)/XDIM_MAX_THREADS ,imgutil->getHeight()-wsize);

	dim3 dimBlock(XDIM_Q_THREADS);
    dim3 dimGrid(ceil((float)imgutil->getWidth() /XDIM_Q_THREADS),imgutil->getHeight()-wsize);

    dim3 argBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 argGrid(ceil((float) imgutil->getWidth() / BLOCK_SIZE),ceil( (float)imgutil->getHeight()/ BLOCK_SIZE));

    int width = imgutil->getWidth();
    int height = imgutil->getHeight();


    int size1 = height*ndisp;
    int size2 = width*ndisp;

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

    cudaStream_t stream1;
    cudaStream_t stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
   

	//#######################################################################################################################//


    for(size_t i=0; i<limg.size(); i++){


    	imgutil->read_image(limg[i],imgl);
		imgutil->read_image(rimg[i],imgr);
		
		
		inti_cost<<< argGrid, argBlock,0,stream1 >>>( cost_d, height,width,ndisp,maxcost );

	    cudaMemcpyAsync( imgl_d, imgl, width*height*sizeof(float), cudaMemcpyHostToDevice,stream1);
	    cudaMemcpyAsync( imgr_d, imgr, width*height*sizeof(float), cudaMemcpyHostToDevice,stream2);


	    if( ((2*XDIM_Q_THREADS+ndisp)*tchuncks)*sizeof(uint64)< SHARED_MEMORY){


	    	CensusTransformKernel<<<dimGridCens, dimBlockCens,XDIM_MAX_THREADS*sizeof(float)>>>(imgl_d,census_l_d,height, width,wsize,tchuncks);
	    	CensusTransformKernel<<<dimGridCens, dimBlockCens,XDIM_MAX_THREADS*sizeof(float)>>>(imgr_d,census_r_d,height, width,wsize,tchuncks);
	   		CensusSADKernel<<<dimGrid, dimBlock,((2*XDIM_Q_THREADS+ndisp)*tchuncks)*sizeof(uint64)>>>(census_l_d,census_r_d,cost_d,height, width,ndisp
	   																									,wsize,maxcost,tchuncks,((XDIM_Q_THREADS+ndisp)*tchuncks));


	    }else{

	   		//Generic no limit. slower
	    	CensusTransformKernelgen<<<dimGridCens, dimBlockCens,XDIM_MAX_THREADS*sizeof(float)>>>(imgl_d,census_l_d,height, width,wsize,tchuncks);
	    	CensusTransformKernelgen<<<dimGridCens, dimBlockCens,XDIM_MAX_THREADS*sizeof(float)>>>(imgr_d,census_r_d,height, width,wsize,tchuncks);
	    	CensusSADKernelgen<<<dimGrid, dimBlock,(XDIM_Q_THREADS+ndisp)*sizeof(uint64)>>>(census_l_d,census_r_d,cost_d,height, width,ndisp,wsize,maxcost,tchuncks);
		}

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