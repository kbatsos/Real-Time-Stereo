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


__global__ void NisterPrecompute(uint64 * Al,uint64 * Ar,double* Cl, double* Cr, 
                                const uint64* l_integral, const uint64* r_integral, const unsigned long long * l_sq_integral,  const unsigned long long* r_sq_integral,
                                const int rows , const int cols, const int wsize, const int wc, const int sqwin ){

    extern __shared__ unsigned long long integralparts[];

    const int Row = threadIdx.x;
    const int Col = blockIdx.x; 
    const int o_index = (Row+wc)*cols+Col+wc;
    const int top = (Row)*4;
    const int bottom = (Row+wsize)*4;
    const int tr = (Row)*cols+Col+wsize;
    const int tl = Row*cols+Col;

    if(Row<(rows+1)&& Col < cols-wsize){
        integralparts[Row*4] =l_integral[tr]-l_integral[tl];
        integralparts[Row*4+1] =r_integral[tr] -r_integral[tl];
        integralparts[Row*4+2] =l_sq_integral[tr] -l_sq_integral[tl];
        integralparts[Row*4+3] =r_sq_integral[tr] -r_sq_integral[tl];
    }


    __syncthreads();

    if(Row < rows-wsize && Col < cols-wsize){
        uint32 al =  integralparts[bottom] - integralparts[top];
        uint32 ar = integralparts[bottom+1] - integralparts[top+1];


        unsigned long long Bl = integralparts[bottom+2]- integralparts[top+2];
        unsigned long long Br = integralparts[bottom+3]- integralparts[top+3];

        Al[o_index] = al;
        Ar[o_index] = ar;


        Cl[ o_index ] = 1/(sqrt(sqwin*Bl - (double)( al )*( al ) ));
        Cr[ o_index ] = 1/(sqrt(sqwin*Br - (double)( ar )*( ar) ));        

    }
    


}



__global__ void NisterMatch(const float* left, const float* right, double* integral_vol,  const int rows, const int cols, const int integrrows , const int integrcols , const int ndisp,const int offset ){

    extern __shared__ float row_slice[];

    int Col =  threadIdx.x+offset;
    int Row = blockIdx.x;

    if(Col <cols && Row<rows ){
        row_slice[threadIdx.x] = left[Row*cols+Col];
        row_slice[blockDim.x+ndisp+threadIdx.x] = right[Row*cols+Col];
        
    }

    float rp = ceil( (float)ndisp/blockDim.x  );

     for(int b=0; b<rp; b++){

        if(blockIdx.x > 0 && (threadIdx.x+b*blockDim.x) < ndisp ){
            row_slice[blockDim.x+(threadIdx.x+b*blockDim.x)] = right[Row*cols+(Col -(ndisp-b*blockDim.x))];
        }
    }    


    __syncthreads();

    for(int d=0; d<ndisp; d++){

        if(Row < rows && Col < cols && Col-d >=0 ){
            integral_vol[d*integrrows*integrcols+Row*integrcols + Col] = row_slice[threadIdx.x] * row_slice[(blockDim.x+ndisp)+threadIdx.x-d];
        }


    }

}

__global__ void NCC(const double* integral_vol, double* slice,
                        const uint64 * Al,const uint64 * Ar,const double* Cl, const double* Cr, 
                        const int integrrows , const int integrcols ,  const int rows , 
                        const int cols, const int wsize, const int wc,const int sqwin,const int ndisp, const int warpwidth){

    extern __shared__  __align__(sizeof(double)) unsigned char ncc_shared[];

    uint64 * Ar_sm = reinterpret_cast<uint64 *>(&ncc_shared[0]);     
    double * Cr_sm = reinterpret_cast<double *>(&ncc_shared[(XDIM_Q_THREADS+ndisp)*sizeof(double)]);  

    const int Row = blockIdx.y;
    const int Col =blockIdx.x*blockDim.x + threadIdx.x;

    int threaddispl = 0;
    if(blockIdx.x >0){
    	threaddispl=ndisp;
    }

    float rp = ceil( (float)ndisp/blockDim.x  );

     for(int b=0; b<rp; b++){

        if(blockIdx.x > 0 && (threadIdx.x+b*blockDim.x) < ndisp ){
            Ar_sm[(threadIdx.x+b*blockDim.x)] = Ar[(Row+wc)*cols +  (Col -(ndisp-b*blockDim.x)+wc) ];
            Cr_sm[(threadIdx.x+b*blockDim.x)] = Cr[(Row+wc)*cols +  (Col -(ndisp-b*blockDim.x)+wc) ];
        }
    }


	if(Row < rows-wsize && Col < cols-wsize){
		const int index = (Row+wc)*cols+(Col+wc);
		const uint64 al = Al[index];
		const double cl = Cl[index];

		Ar_sm[threaddispl+ threadIdx.x ] = Ar[index ];
		Cr_sm[threaddispl+ threadIdx.x ] = Cr[index ];

		__syncthreads();




	   	#pragma unroll
	    for (int d=0; d< ndisp; d++){
	    
		    const int dindex = threaddispl+threadIdx.x-d;
		    const int disp = d*integrrows*integrcols;
		    double ncccost = 2;


		    if(Col < cols-wsize && dindex >=0){

		            const double lD = integral_vol[disp+(Row+wsize)*integrcols + (Col+wsize)]
		                                            - integral_vol[disp+(Row)*integrcols + (Col+wsize)]
		                                            - integral_vol[disp+(Row+wsize)*integrcols + Col]
		                                            + integral_vol[disp+Row*integrcols + Col];


		            if( isfinite(cl) &&  isfinite(Cr_sm[ dindex ])){                                
		            	ncccost = 1-((double)(sqwin*lD- al * Ar_sm[dindex] )*cl*Cr_sm[ dindex ]) ;   
		            }                             
		        
		    }

		        slice[d*rows*cols+(Row+wc)*cols + (Col+wc)] = ncccost;
		}

	}
}




__global__ void square( const float* input, uint64 * output,  const int height, const int width,const int oHeigth, const int oWidth){

	const int index = blockIdx.x*blockDim.x+threadIdx.x;

 	if(index< height*width){
		uint64 val = (uint64)input[index];

		val = val*val;
		__syncthreads();

		output[index] = val;

	}
}

void usage(void){
    std::cout   << "NCC genmeric CUDA implementation" << std::endl;
    std::cout   << "Arguments" << std::endl;
    std::cout   << "-l:\t\t Left image  | File containing names of the left images" << std::endl;
    std::cout   << "-r:\t\t Right image | File containing the names of the right images" << std::endl;
    std::cout   << "-ndisp:\t\t Number of Disparities" << std::endl;
    std::cout   << "-wsize:\t\t Window size" << std::endl; 
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
    int wsize=9; 
    int ndisp=256;
    bool post=false;

    bool single=true;

    int argsassigned = 0;
    int required=0;

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
    const int warpwidth = wdiv*32;

	cudaStream_t stream1;
    cudaStream_t stream2;
 
    //cudaError_t strerr;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    const int wc = wsize/2;
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
    //missing kernel
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

    int vthreads = XDIM_MAX_THREADS;

    if(height < XDIM_Q_THREADS)
        vthreads=XDIM_Q_THREADS;
    else if(height < XDIM_Q_THREADS)
        vthreads=XDIM_H_THREADS;


    int vreps = ceil((float) height/vthreads );
 
    dim3 integraldim2Grid(1,width,1 );
    dim3 integraldim1Grid(1,height,1 );

    dim3 integraldim1Griddisp(1,height,ndisp );

    int hthreads = XDIM_MAX_THREADS;

    if(width < XDIM_Q_THREADS)
        width=XDIM_Q_THREADS;
    else if(width<XDIM_H_THREADS)
        width=XDIM_H_THREADS;

    int hreps = ceil((float)width/hthreads);

    dim3 hintegralGrid(height,1, 1 );

    dim3 preCompBlock(XDIM_MAX_THREADS);
    dim3 preCompGrid(width);

    dim3 MatchGrid(height);

    dim3 vintegralGriddisp(1,width,ndisp );


    dim3 swapBlock(BLOCK_D_SIZE,16,1);
    dim3 swapGrid(ceil((float)imgutil->getWidth()*imgutil->getHeight()/BLOCK_D_SIZE),ceil((float) ndisp/BLOCK_D_SIZE ));

    dim3 transBlock(BLOCK_D_SIZE,16,1);
    dim3 transGrid(ceil((float)imgutil->getWidth()/BLOCK_D_SIZE ),ceil((float)imgutil->getHeight()/BLOCK_D_SIZE));

    dim3 transinvGrid(ceil((float)imgutil->getHeight()/BLOCK_D_SIZE),ceil((float)imgutil->getWidth()/BLOCK_D_SIZE ));

    dim3 argBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 argGrid(ceil((float) imgutil->getWidth() / BLOCK_SIZE),ceil( (float)imgutil->getHeight()/ BLOCK_SIZE));

    dim3 argBlockSq(XDIM_MAX_THREADS);
    dim3 argGridSq(ceil((float)imgutil->getWidth()*imgutil->getHeight()/(XDIM_MAX_THREADS)));

    dim3 dimBlockNCC(XDIM_Q_THREADS);
    dim3 dimGridNCC(ceil((float) imgutil->getWidth() / XDIM_Q_THREADS),imgutil->getHeight()-wsize);
    //########################################################################################################################################//



    for(size_t i=0; i<limg.size(); i++){



            imgutil->read_image(limg[i],imgl);
            imgutil->read_image(rimg[i],imgr);

    	    cudaMemcpyAsync( imgl_d, imgl, width*height*sizeof(float), cudaMemcpyHostToDevice,stream1);
    	    cudaMemcpyAsync( imgr_d, imgr, width*height*sizeof(float), cudaMemcpyHostToDevice,stream2);

    	    
    	    cudaMemsetAsync(cost_d,2 , height*width*ndisp*sizeof(double),stream1);
    	    cudaMemsetAsync(post_cost_d,2 , width*height*ndisp*sizeof(double),stream2);


            for (int r =0; r<hreps; r++){
               int tthreads=hthreads;
               if(r >0 && r==hreps-1)
                    tthreads = width-r*(hthreads-1); 

        	   HorizontalIntegralKernel_outofplace<<<hintegralGrid, tthreads, tthreads*sizeof(uint64) >>>(l_integral_d, imgl_d, height ,  width , 1,r*(hthreads-1));
        	   HorizontalIntegralKernel_outofplace<<<hintegralGrid, tthreads, tthreads*sizeof(uint64) >>>(r_integral_d, imgr_d, height ,  width , 1,r*(hthreads-1));
            }

        	transpose<<< transGrid, transBlock >>>( l_integral_d, l_integral_d_t,height,width);
        	transpose<<< transGrid, transBlock >>>( r_integral_d, r_integral_d_t,height,width);

            for(int r=0;r<vreps;r++){
                int tthreads=vthreads;
                if(r >=0 && r==vreps-1)
                    tthreads=height-r*(vthreads-1);

        	   IntegralKernel<<<integraldim2Grid, tthreads, 32*sizeof(uint64) >>>(l_integral_d_t,width, height,1,r*(vthreads-1) );
        	   IntegralKernel<<<integraldim2Grid, tthreads, 32*sizeof(uint64) >>>(r_integral_d_t,width, height,1,r*(vthreads-1) );
            }

        	transpose<<< transinvGrid, transBlock >>>(l_integral_d_t, l_integral_d, width,height);
        	transpose<<< transinvGrid, transBlock >>>(r_integral_d_t, r_integral_d, width,height);


        	square<<< argGridSq,argBlockSq >>>( imgl_d, l_sq_integral_d, height,width, height, width);
        	square<<< argGridSq,argBlockSq >>>( imgr_d, r_sq_integral_d, height,width, height, width);



            for (int r =0; r<hreps; r++){
               int tthreads=hthreads;
               if(r >0 && r==hreps-1)
                    tthreads = width-r*(hthreads-1); 

        	   IntegralKernel<<<integraldim1Grid, tthreads, 32*sizeof(uint64) >>>(l_sq_integral_d, height, width,1,r*(hthreads-1) );
        	   IntegralKernel<<<integraldim1Grid, tthreads, 32*sizeof(uint64) >>>(r_sq_integral_d, height, width,1,r*(hthreads-1) );

            }

        	transpose<<< transGrid, transBlock >>>( l_sq_integral_d, l_integral_d_t,height,width);
        	transpose<<< transGrid, transBlock >>>( r_sq_integral_d, r_integral_d_t,height,width);

            for(int r=0;r<vreps;r++){
                int tthreads=vthreads;
                if(r >=0 && r==vreps-1)
                    tthreads=height-r*(vthreads-1);            

        	   IntegralKernel<<<integraldim2Grid, tthreads, 32*sizeof(uint64) >>>(l_integral_d_t,width, height,1,r*(vthreads-1) );
        	   IntegralKernel<<<integraldim2Grid, tthreads, 32*sizeof(uint64) >>>(r_integral_d_t,width, height,1,r*(vthreads-1) );

            }

        	transpose<<< transinvGrid, transBlock >>>( l_integral_d_t, l_sq_integral_d,width,height);
        	transpose<<< transinvGrid, transBlock >>>( r_integral_d_t, r_sq_integral_d,width,height);



        	NisterPrecompute<<< preCompGrid,preCompBlock,XDIM_MAX_THREADS*4*sizeof(unsigned long long int) >>>(Al_d, Ar_d, Cl_d, Cr_d, 
                                                        l_integral_d , r_integral_d , l_sq_integral_d, r_sq_integral_d, 
                                                        height , width,wsize,wc,wsize*wsize );

            for (int r =0; r<hreps; r++){
               int tthreads=hthreads;
               if(r >0 && r==hreps-1)
                    tthreads = width-r*hthreads; 


        	   NisterMatch<<<MatchGrid, tthreads, (2*tthreads+ndisp)*sizeof(float)>>>(imgl_d,imgr_d,post_cost_d,height , width,height,width,ndisp,r*hthreads);
            }


            for (int r =0; r<hreps; r++){
               int tthreads=hthreads;
               if(r >0 && r==hreps-1)
                    tthreads = width-r*(hthreads-1); 

        	   IntegralKernel<<<integraldim1Griddisp, tthreads, 32*sizeof(uint64) >>>(post_cost_d, height, width,ndisp,r*(hthreads-1) );
            }


            for(int r=0;r<vreps;r++){
                int tthreads=vthreads;
                if(r >=0 && r==vreps-1)
                    tthreads=height-r*(vthreads-1); 

        	   VerticalIntegralKernel<<<vintegralGriddisp, tthreads,32*sizeof(double) >>>(post_cost_d,height,width,1,r*(vthreads-1));
            }

    		NCC<<<dimGridNCC, dimBlockNCC,2*(XDIM_Q_THREADS+ndisp)*sizeof(double)>>>(post_cost_d,cost_d,
    		                                                                          Al_d, Ar_d, Cl_d, Cr_d,
    		                                                                          height,width,height,width ,wsize,wc,wsize*wsize,ndisp,warpwidth);
    		   

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