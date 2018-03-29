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
#include <pgm.h>

using namespace std;
using namespace std::chrono;
typedef uint8_t uint8;
typedef unsigned int uint;
typedef unsigned long long ull;
typedef unsigned long long int uint64;



void nccNister(uint8* leftp, uint8 *rightp, double* cost, int* shape, int ndisp, int wsize){




    int end;
    int wc = wsize/2;
    int sqwin = wsize*wsize;

    uint64* Al = (uint64 *)calloc(shape[0]*shape[1],sizeof(uint64));
    uint64* Ar = (uint64 *)calloc(shape[0]*shape[1],sizeof(uint64));
    double* Cl = (double *)calloc(shape[0]*shape[1],sizeof(double));
    double* Cr = (double *)calloc(shape[0]*shape[1],sizeof(double));

    for (int i=0; i< shape[0]-wsize;i++){
    	for(int j=0; j< shape[1]-wsize; j++){
    		double Bl =0;
    		double Br =0;
    		for (int wh=0; wh<wsize;wh++){
    		    for(int ww=0; ww<wsize;ww++){

    		    	Al[(i+wc)*shape[1]+(j+wc)] += leftp[(i+wh)*shape[1] + (j+ww) ];
    		    	Ar[(i+wc)*shape[1]+(j+wc)] += rightp[(i+wh)*shape[1] + (j+ww) ];
    		    	Bl += leftp[(i+wh)*shape[1] + (j+ww) ]*leftp[(i+wh)*shape[1] + (j+ww) ];
    		    	Br += rightp[(i+wh)*shape[1] + (j+ww) ]*rightp[(i+wh)*shape[1] + (j+ww) ];

    		    }
    		}

    		Cl[ (i+wc)*shape[1]+(j+wc) ] = 1/(sqrt(sqwin*Bl - (double)( Al[(i+wc)*shape[1]+(j+wc)] )*( Al[(i+wc)*shape[1]+(j+wc)] ) ));
    		if(!std::isfinite(Cl[ (i+wc)*shape[1]+(j+wc)]))
    			Cl[ (i+wc)*shape[1]+(j+wc) ]=0;
    		Cr[ (i+wc)*shape[1]+(j+wc) ] = 1/(sqrt(sqwin*Br - (double)( Ar[(i+wc)*shape[1]+(j+wc)] )*( Ar[(i+wc)*shape[1]+(j+wc)] ) ));
    		if(!std::isfinite(Cr[ (i+wc)*shape[1]+(j+wc)]))
    			Cr[ (i+wc)*shape[1]+(j+wc) ]=0;
    	}
    }


    for (int i=0; i< shape[0]-wsize; i++){
    	for(int j=0; j< shape[1]-wsize; j++){

    		end = std::min(ndisp,j+1);
    		for(int d=0; d < end; d++){

    			double D =0;

    			for (int wh=0; wh<wsize;wh++){
    				for(int ww=0; ww<wsize;ww++){


    					D +=  leftp[(i+wh)*shape[1] + (j+ww) ] * rightp[(i+wh)*shape[1] + (j-d+ww) ];
    				}
    			}



    			cost[(i+wc)*shape[1]*ndisp + (j+wc)*ndisp+ d] = -(double)(sqwin*D- Al[(i+wc)*shape[1]+(j+wc)] * Ar[(i+wc)*shape[1]+(j-d+wc)] )*Cl[ (i+wc)*shape[1]+(j+wc) ]*Cr[ (i+wc)*shape[1]+(j-d+wc) ];
    		}
    	}
    }

    delete [] Al;
    delete [] Ar;
    delete [] Cl;
    delete [] Cr;

}



void argmin( double*cost, float* disp,int* shape ){

	for(int i=0; i< shape[0]; i++){
		for(int j=0; j<shape[1]; j++){
			float cur_d = 0;
			float cur_c = cost[i*shape[1]*shape[2]+j*shape[2]];
			for(int d=1; d<shape[2]; d++){
				float c_v = cost[i*shape[1]*shape[2]+j*shape[2]+d];
				if( c_v < cur_c ){
					cur_c = c_v;
					cur_d = d;
				}
			}
			disp[i*shape[1]+j] = cur_d;

		}
	}


}


void usage(void){
	std::cout<< "NCC vanilla implementation" << std::endl;
	std::cout<< "Arguments" << std::endl;
	std::cout<< "-l:\t File containing the names of the left images" << std::endl;
	std::cout << "-r:\t File containing the names of the right images" << std::endl;
	std::cout << "-ndisp:\t Number of Disparities" << std::endl;
	std::cout << "-wsize:\t Window size" << std::endl; 
}

std::vector<string> getImages(string file){
	    std::vector<string> imageNames;

        std::ifstream ifs(file.c_str());
        std::string line;
        if(ifs.is_open()){
                while(std::getline(ifs,line )){
                        imageNames.push_back(line);
                }

        }else{
                std::cout << "File " << file << " does not exist! " <<std::endl;
                exit(0);
        }

        return imageNames;

}


int main(int argc,char* argv[]){

	high_resolution_clock::time_point t1;
	high_resolution_clock::time_point t2;

	string leftfile;
	string rightfile;
	int ndisp=256;
	int wsize=3; 

	int argsassigned = 0;

	for(int i=0; i<argc; i++){
		if( !strcmp(argv[i], "-l") ){
			//sprintf(leftImages, "%s", argv[++i]);
			leftfile = string(argv[++i]);
			argsassigned++;
		}else if( !strcmp(argv[i],"-r") ){
			//sprintf(rightImages, "%s", argv[++i]);
			rightfile = string(argv[++i]);
			argsassigned++;
		}else if( !strcmp(argv[i],"-ndisp") ){
			ndisp= atoi(argv[++i]);
			argsassigned++;
		}else if( !strcmp(argv[i],"-wsize") ){
			wsize= atoi(argv[++i]);
			argsassigned++;
		}
	}

	if(argsassigned ==0 || argsassigned ==1){
		leftfile = string("../../leftimg.txt");
		rightfile = string("../../rightimg.txt");
	}
	else if( argsassigned < 4 ){
		usage();
		return 0;
	}		

	std::vector<string> limg = getImages(leftfile);
	std::vector<string> rimg = getImages(rightfile);


	pgm* imgutil = new pgm();

	//Allocate stuff on the device 
	imgutil->read_png_file(limg[0].c_str());

	uint8* imgl= (uint8*)calloc(imgutil->getWidth()*imgutil->getHeight(),sizeof(uint8));
	uint8* imgr= (uint8*)calloc(imgutil->getWidth()*imgutil->getHeight(),sizeof(uint8));
	float* disp= (float*)calloc(imgutil->getWidth()*imgutil->getHeight(),sizeof(float));

	imgutil->read_png_file(limg[0].c_str(),imgl);
	imgutil->read_png_file(rimg[0].c_str(),imgr);

	double* cost=(double*)calloc(imgutil->getWidth()*imgutil->getHeight()*ndisp,sizeof(double));

	int* shape = new int[3];
	shape[0] = imgutil->getHeight();shape[1] = imgutil->getWidth(),shape[2]=ndisp;

	t1 = high_resolution_clock::now();
	nccNister(imgl, imgr, cost, shape, ndisp, wsize);

	argmin( cost,  disp, shape );

	t2 = high_resolution_clock::now();
 	auto duration = duration_cast<microseconds>( t2 - t1 ).count();
 	std::cout << duration/1000 << std::endl; 	

	imgutil->writeDisparityMap("disp.png",disp);


	return 0;
}