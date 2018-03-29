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


void census(uint8* leftp, uint8 *rightp, uint* cost, int* shape, int ndisp, int wsize){
  

    int end;
    int wc = wsize/2;
    int sqwin = wsize*wsize;

    bool* censustl = new bool[shape[0]*shape[1]*sqwin];
    bool* censustr = new bool[shape[0]*shape[1]*sqwin];

    for (int i=0; i< shape[0]-wsize;i++){
    	for(int j=0; j< shape[1]-wsize; j++){

    		for (int wh=0; wh<wsize;wh++){
    		    for(int ww=0; ww<wsize;ww++){

    		    	censustl[ (i+wc)*shape[1]*sqwin + (j+wc)*sqwin +wh*wsize+ww ] = (leftp[ (i+wc)*shape[1] + (j+wc) ] < leftp[(i+wh)*shape[1] + (j+ww) ]);
    		    	censustr[ (i+wc)*shape[1]*sqwin + (j+wc)*sqwin +wh*wsize+ww ] = (rightp[ (i+wc)*shape[1] + (j+wc) ] < rightp[(i+wh)*shape[1] + (j+ww) ]);
    		    }
    		}

    	}
    }


    for (int i=0; i< shape[0]-wsize; i++){
    	for(int j=0; j< shape[1]-wsize; j++){

    		end = std::min(ndisp,j+1);
    		for(int d=0; d < end; d++){

    			int sum =0;

    			for (int wh=0; wh<wsize;wh++){
    				for(int ww=0; ww<wsize;ww++){


    					sum += std::abs( censustl[(i+wc)*shape[1]*sqwin + (j+wc)*sqwin +wh*wsize+ww ] - censustr[ (i+wc)*shape[1]*sqwin + (j-d+wc)*sqwin +wh*wsize+ww ] );
    				}
    			}

    			cost[(i+wc)*shape[1]*ndisp + (j+wc)*ndisp+ d] = sum;
    		}
    	}
    }

    delete [] censustl;
    delete [] censustr;

}


void argmin( uint*cost, float* disp,int* shape ){

	for(int i=0; i< shape[0]; i++){
		for(int j=0; j<shape[1]; j++){
			float cur_d = 0;
			uint cur_c = cost[i*shape[1]*shape[2]+j*shape[2]];
			for(int d=1; d<shape[2]; d++){
				uint c_v = cost[i*shape[1]*shape[2]+j*shape[2]+d];
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

	uint* cost=(uint*)calloc(imgutil->getWidth()*imgutil->getHeight()*ndisp,sizeof(uint));


	int* shape = new int[3];
	shape[0] = imgutil->getHeight();shape[1] = imgutil->getWidth(),shape[2]=ndisp;
	std::fill_n(cost,shape[0]*shape[1]*ndisp, wsize*wsize-1);

	t1 = high_resolution_clock::now();
	census(imgl, imgr, cost, shape, ndisp, wsize);

	argmin( cost,  disp, shape );

	t2 = high_resolution_clock::now();
 	auto duration = duration_cast<microseconds>( t2 - t1 ).count();
 	std::cout <<  duration/1000 << std::endl; 	

	imgutil->writeDisparityMap("disp.png",disp);


	return 0;
}