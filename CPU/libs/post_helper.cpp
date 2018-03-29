#include "StereoBMHelper.h"
#include "FastFilters.h"
#include "StereoSGM.h"


template<typename T>
uint16* prepare_cost(T* cost,int* shape,float scale,int offset){

	
	uint16* dsi = (uint16*)_mm_malloc(shape[0]*shape[1]*shape[2]*sizeof(uint16), 32);

#pragma omp parallel num_threads(12)
	{
	    #pragma omp for
    	for(int i=0; i<shape[0];i++ ){
    		for(int j=0; j< shape[1]; j++){
				for(int d=0; d<shape[2]; d++){

					dsi[i*shape[1]*shape[2]+j*shape[2]+d] = (uint16)((cost[d*shape[0]*shape[1]+i*shape[1]+j]+offset)*scale);
				}
			}
		}
	}

	return dsi;
}


void parseConf(int &numstrips,StereoSGMParams_t &params,std::string conf ){
		const std::string& chars = "\t\n\v\f\r ";

	    std::ifstream ifs(conf.c_str());
        std::string line;
        if(ifs.is_open()){
                while(std::getline(ifs,line )){
                	   std::string opt = line.substr(0,line.find_last_of(":"));
                	   opt.erase(0, opt.find_first_not_of(chars));
                	   opt.erase(opt.find_last_not_of(chars) + 1);

                	   int start = line.find_last_of(":")+1;
                	   int end =  line.find_first_of("#") - start;

                	   std::string val = line.substr(start,end);

                	   val.erase(0, val.find_first_not_of(chars));
                	   val.erase(val.find_last_not_of(chars) + 1);

                	   if(!strcmp(opt.c_str(),"numStrips")){
                	   		numstrips = atoi(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"InvalidDispCost")){
                	   		params.InvalidDispCost = atoi(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"lrCheck")){
                	   		params.lrCheck = atoi(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"MedianFilter")){
                	   		params.MedianFilter = atoi(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"Paths")){
                	   		params.Paths = atoi(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"subPixelRefine")){
                	   		params.subPixelRefine = atoi(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"NoPasses")){
                	   		params.NoPasses = atoi(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"rlCheck")){
                	   		params.rlCheck = atoi(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"P1")){
                	   		params.P1 = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"P2min")){
                	   		params.P2min = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"Alpha")){
                	   		params.Alpha = atof(val.c_str());
                	   }else if(!strcmp(opt.c_str(),"Gamma")){
                	   		params.Gamma = atof(val.c_str());
                	   }  
                }

        }else{
                std::cout << "File " << conf << " does not exist! " <<std::endl;
                exit(0);
        }
}

template<typename T>
void argmin( T*cost, float* disp,int* shape ){

	#pragma omp parallel for
	for(int i=0; i< shape[0]; i++){
		for(int j=0; j<shape[1]; j++){
			float cur_d = 0;
			float cur_c = cost[i*shape[1]+j];
			for(int d=1; d<shape[2]; d++){
				float c_v = cost[d*shape[0]*shape[1] +i*shape[1]+j];
				if( c_v < cur_c ){
					cur_c = c_v;
					cur_d = d;
				}
			}
			disp[i*shape[1]+j] = cur_d;

		}
	}


}


void correctEndianness(uint16* input, uint16* output, uint32 size)
{
    uint8* outputByte = (uint8*)output;
    uint8* inputByte = (uint8*)input;
#pragma omp parallel num_threads(12)
    {
    	#pragma omp for
    	for (uint32 i=0; i < size; i++) {
        	*(outputByte+1) = *inputByte;
        	*(outputByte) = *(inputByte+1);
        	outputByte+=2;
        	inputByte+=2;
    	}
	}
}

template<typename T>
void doPost( T*cost, int* shape,uint8* refImg,string name,string out_t,float scale,int offset, int numStrips, StereoSGMParams_t params ){

	imgio* imgutil = new imgio();
	imgutil->setHeight(shape[0]);
	imgutil->setWidth(shape[1]);
	
    
    uint16* refimg = (uint16*)_mm_malloc(shape[0]*shape[1]*sizeof(uint16), 16);
    uint16* dsi =  prepare_cost(cost,shape,scale,offset);
    

#pragma omp parallel num_threads(12)
    {
		#pragma omp for
    	for(int i=0; i<shape[0]*shape[1];i++ ){
			refimg[i] = refImg[i];
		}
	
	}


	float32* dispImg = (float32*)_mm_malloc(shape[0]*shape[1]*sizeof(float32), 16);
    float32* dispImgRight = (float32*)_mm_malloc(shape[0]*shape[1]*sizeof(float32), 16);

    uint16* leftImg = (uint16*)_mm_malloc(shape[0]*shape[1]*sizeof(uint16), 16);

 	correctEndianness(refimg, leftImg, shape[0]*shape[1]);


	StripedStereoSGM<uint16> stripedStereoSGM(shape[1], shape[0], shape[2]-1, numStrips, 4, params);


	stripedStereoSGM.process(leftImg, dispImg, dispImgRight,dsi, numStrips,1);

	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0; i< shape[0]*shape[1]; i++){
			if(dispImg[i]<0)
				dispImg[i]=0;		
		}
	}

	imgutil->write_image(name,dispImg,out_t);

 	_mm_free(dsi);
	delete imgutil;

}



uint8* paddwidth( uint8* img, int height, int width ){

	int pad = 16 - width%16;
	int widthp = width+pad;

	uint8* imgp= (uint8*)calloc(height*(width+pad),sizeof(uint8));

	#pragma omp parallel
	{
		#pragma omp for
		for(int i=0; i< height; i++){
			for(int j=0; j<width;j++){
				imgp[i*widthp+j] = img[i*width+j];

			}

			for(int j=0; j<pad; j++){
				imgp[i*widthp +j] = img[i*width+(width-1)];
			}
		}
	}


	free(img);
	return imgp;



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