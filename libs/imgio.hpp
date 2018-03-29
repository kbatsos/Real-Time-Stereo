/*
 * pgm.h
 *
 *  Created on: Jun 9, 2016
 *      Author: kochoba
 */

#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <stdint.h>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <stdarg.h>
#include <png++/png.hpp>
#include <png.h>


#ifndef PGM_H_
#define PGM_H_

typedef uint16_t uint16;
typedef unsigned char uchar;
typedef uint8_t uint8;

class imgio {
public:
	imgio(){
		this->height = 0;
		this->width=0;
		this->intensity=255;
	}

	void read_image_meta( std::string filename ){
		std::string ext = filename.substr(filename.find_last_of(".")+1);
		if(ext == "png"){
			this->read_png_file(filename.c_str());
		}else if (ext == "pgm"){
			this->readpgm(filename.c_str());
		}else if(ext == "pfm"){
			this->readpfm(filename.c_str());
		}else{
			std::cout << "Unsupported image extension exiting..." << std::endl;
			exit(0);
		}
	}
	template<typename T>
	void read_image( std::string filename,T* img ){
		std::string ext = filename.substr(filename.find_last_of(".")+1);
		if(ext == "png"){
			this->read_png_file(filename.c_str(),img);
		}else if (ext == "pgm"){
			this->readpgm(filename.c_str(),img);
		}else if(ext == "pfm"){
			this->readpfm(filename.c_str(),img);
		}else{
			std::cout << "Unsupported image extension! exiting..." << std::endl;
			exit(0);
		}
	}	

	template<typename T>
	void write_image(const std::string file_name,T* data_,std::string out_t){
		std::string fname = file_name.substr(0,file_name.find_last_of(".")+1);
		if(out_t == "pgm"){
			this->writepgm((fname+"pgm").c_str(),data_);
		}else if(out_t=="pfm"){
			this->writepfm((fname+"pfm").c_str(),data_);
		}else if(out_t=="png"){
			this->write_png_file((fname+"png").c_str(),data_);
		}else if(out_t=="disp"){
			this->writeDisparityMap ((fname+"png").c_str(),data_);
		}
		else{
			std::cout << "Unsupported output type! exiting..." << std::endl;
			exit(0);
		}

	}

   
   void readpgm(const char* filename){
  	std::ifstream ifs(filename,std::ifstream::binary);
  	std::string c;

		if (ifs){
			ifs >> c;
			if(!c.compare("P2")|| !c.compare("P5")){
				ifs >> c;
				if(c.substr(0,1).compare("#") == 0){
					getline(ifs,c);
					ifs >> c;
				}
				this->width = atoi(c.c_str());
				ifs >> c;
				this->height = atoi(c.c_str());
				ifs >> c;
				this->intensity = atoi(c.c_str());
				char b;
				ifs.read(&b, 1);


			}else{
				std::cout << "Invalid magic number!" << std::endl;
				ifs.close();
				exit(1);
			}

		}else{
			std::cout << "File " << filename << " does not exist!" << std::endl;
			ifs.close();
			exit(1);
		}
		ifs.close();

  }	

	template<typename T>
	void readpgm(const char* filename,T* img){
		std::ifstream ifs(filename,std::ifstream::binary);
		std::string c;

		if (ifs){
		ifs >> c;
			if(!c.compare("P2")|| !c.compare("P5")){
				ifs >> c;
				if(c.substr(0,1).compare("#") == 0){
					getline(ifs,c);
					ifs >> c;
				}
				this->width = atoi(c.c_str());
				ifs >> c;
				this->height = atoi(c.c_str());
				ifs >> c;
				this->intensity = atoi(c.c_str());
				char b;
				ifs.read(&b, 1);
				unsigned char * buffer = new unsigned char[this->width*this->height];
				ifs.read((char *)buffer,this->width*this->height);

				for(int i=0; i< this->width*this->height; i++)
					img[i] = (T) buffer[i];

				delete[] buffer;
				if(ifs){
				}
				else{
					std::cout << "Error only "<< ifs.gcount()<< " was read..." << std::endl;
					ifs.close();
					exit(1);
				}


			}else{
				std::cout << "Invalid magic number!" << std::endl;
				ifs.close();
				exit(1);
			}

		}else{
			std::cout << "File " << filename << " does not exist!" << std::endl;
			ifs.close();
			exit(1);
		}
		ifs.close();

	}
	template<typename T>
	void writepgm(const char* filename,T* imgbuffer){
		char* buff = (char*)calloc(this->width*this->height,sizeof(char));
		
		#pragma omp parallel
		{
			#pragma omp for
			for (int i=0; i<this->width*this->height;i++ ){
				buff[i] = (char)imgbuffer[i];
			}
		}

		std::ofstream ofs(filename,std::ifstream::binary);
		ofs << "P5\n" << this->width << " " << this->height << "\n"<<255 << "\n";
		ofs.write((char *)buff,this->width*this->height);
		ofs.close();
		delete[] buff;
	}


	void readpfm(const char* filename){
		FILE * pFile;
		pFile = fopen(filename,"rb");
		char c[100];

		if (pFile != NULL){
			int res = fscanf(pFile, "%s", c);
			if(res!=EOF && !strcmp(c,"Pf")){
				res = fscanf(pFile, "%s", c);
				this->width = atoi(c);
				res =fscanf(pFile, "%s", c);
				this->height = atoi(c);
				//fscanf(pFile, "%s", c);
				res = fscanf(pFile, "%s", c);
				this->endianess = atof(c);

			}else{
				std::cout << "Invalid magic number! " <<std::endl;
				fclose(pFile);exit(1);
			}

		}else{
			std::cout << "File " << filename << " does not exist!" << std::endl;
			fclose(pFile);exit(1);
		}
		fclose(pFile);

	}
	template<typename T>
	void readpfm(const char* filename,T* buffer){
		FILE * pFile;
		pFile = fopen(filename,"rb");
		char c[100];

		if (pFile != NULL){
			int res = fscanf(pFile, "%s", c);
			if(res !=EOF && !strcmp(c,"Pf")){
				res = fscanf(pFile, "%s", c);
				this->width = atoi(c);
				res =fscanf(pFile, "%s", c);
				this->height = atoi(c);
				res = fscanf(pFile, "%s", c);
				this->endianess = atof(c);
				fseek (pFile , 0, SEEK_END);
				long lSize = ftell (pFile);
				long pos = lSize - this->width*this->height*4;
				fseek (pFile , pos, SEEK_SET);
				T* img = new T[this->width*this->height];
				size_t result =fread(img,sizeof(T),this->width*this->height,pFile);
				fclose(pFile);
				
				//PFM SPEC image stored bottom -> top reversing image
				if(result >0){
				#pragma omp parallel
					{
						#pragma omp for
						for(int i =0; i< this->height; i++){
							 memcpy(&buffer[(this->height -i-1)*this->width],&img[(i*this->width)],this->width*sizeof(T));
						}
					}

					delete [] img;
				}
			}else{
				std::cout << "Invalid magic number! " <<std::endl;
				fclose(pFile);exit(1);
			}

		}else{
			std::cout << "File " << filename << " does not exist!" << std::endl;
			fclose(pFile);exit(1);
		}
		fclose(pFile);

}


	template<typename T>
	void writepfm(const char* filename,T* imgbuffer){
		std::ofstream ofs(filename,std::ifstream::binary);
		ofs << "Pf\n" << this->width << " " << this->height << "\n"<<-1.0<< "\n";
		T* tbimg = (T *)malloc(this->width*this->height*sizeof(T));
		//PFM SPEC image stored bottom -> top reversing image
		#pragma omp parallel
		{
			#pragma omp for
			for(int i =0; i< this->height; i++){
				 memcpy(&tbimg[(this->height -i-1)*this->width],&imgbuffer[(i*this->width)],this->width*sizeof(T));
			}
		}

		ofs.write(( char *)tbimg,this->width*this->height*sizeof(T));
		ofs.close();
		free(tbimg);
	}


	template<typename T>
	void readpbm(const char* filename,T* buffer){
		std::ifstream ifs(filename,std::ifstream::binary);
		std::string c;

		if (ifs){
			ifs >> c;
			if(!c.compare("P4")){
				ifs >> c;
				if(c.substr(0,1).compare("#") == 0){
					getline(ifs,c);
					ifs >> c;
				}
				this->width = atoi(c.c_str());
				ifs >> c;
				this->height = atoi(c.c_str());
				char b;
				ifs.read(&b, 1);
				
				unsigned char c2;
				int numbyte =0;
				int index =0;

				  for ( int j = 0; j < this->height; j++ )
				  {
				    for ( int i = 0; i < this->width; i++ )
				    {
				      if ( i%8 == 0 )
				      {
				        ifs.read ( &b, 1 );
				        c2 = ( unsigned char ) b;
				        if ( ifs.eof ( ) )
				        {
				          std::cout << "\n";
				          std::cout << "PBM - Fatal error!\n";
				          std::cout << "  Failed reading byte " << numbyte << "\n";
				          ifs.close();
				          exit(1);
				        }
				        numbyte = numbyte + 1;
				      }

				      int k = 7 - i%8;
				      int bit = ( c2 >> k )%2;

				      buffer[j*this->width+i] = ((int)bit +1)%2;
				      index = index + 1;
				    }
				  }

				  ifs.close();

			}else if(!c.compare("P1")){
				std::cout << "Not implemented yet" << std::endl;
			}else{
				std::cout << "Invalid magic number!" << std::endl;
				ifs.close();
				exit(1);
			}

		}else{
			std::cout << "File " << filename << " does not exist!" << std::endl;
			ifs.close();
			exit(1);
		}
		ifs.close();
	}

	void abort_(const char * s, ...)
	{
	        va_list args;
	        va_start(args, s);
	        vfprintf(stderr, s, args);
	        fprintf(stderr, "\n");
	        va_end(args);
	        abort();
	}

  
   void read_png_file(const char* file_name){
  	png::image<png::gray_pixel> img(file_name);

  	this->width = img.get_width();
  	this->height = img.get_height();


  }	
  	template<typename T>
    void read_png_file(const char* file_name,T* buffer){


	  	png::image<png::gray_pixel> img(file_name);

	  	this->width = img.get_width();
	  	this->height = img.get_height();


	  	#pragma omp parallel
	  	{
	     #pragma omp for
	     for(uint i=0;i<img.get_height();i++)
		 {
			for(uint j=0;j<img.get_width();j++)
			{
			    buffer[i*this->width+j] = (T)img.get_pixel(j,i);
			}
		 } 
		}


  }	


	template<typename T>
	void write_png_file(const char* file_name,T * buffer){
		png::image<png::gray_pixel> img(this->width,this->height);
	  	#pragma omp parallel
	  	{
		  #pragma omp for
		  for(uint i=0;i<img.get_height();i++)
		  {
		   	for(uint j=0;j<img.get_width();j++)
		    {
		         img.set_pixel(j,i,buffer[i*this->width+j]);
		    }
		  }
		}

		img.write(file_name);
	}


  // is disparity at given pixel to invalid
  inline void setInvalid (float* data_,const int32_t u,const int32_t v) {
    data_[v*this->width+u] = -1;
  }

  // set disparity at given pixel
  inline void setDisp (float* data_,const int32_t u,const int32_t v,const float val) {
    data_[v*this->width+u] = val;
  }

  float* readDisparityMap (const std::string file_name) {
    png::image<png::gray_pixel_16> image(file_name);
    this->width  = image.get_width();
    this->height = image.get_height();
    float* data_   = (float*)malloc(this->width*this->height*sizeof(float));
    #pragma omp parallel
    {
     #pragma omp for
     for (int32_t v=0; v<this->height; v++) {
       for (int32_t u=0; u<this->width; u++) {
        uint16_t val = image.get_pixel(u,v);
        if (val==0) 
        	setInvalid(data_,u,v);
        else        
        	setDisp(data_,u,v,((float)val)/256.0);
       }
     }
	}

    return data_;
  }

    // get disparity at given pixel
  inline float getDisp (float* data_,const int32_t u,const int32_t v) {
    return data_[v*this->width+u];
  }

    // is disparity valid
  inline bool isValid (float* data_,const int32_t u,const int32_t v) {
    return data_[v*this->width+u]>=0;
  }

    void writeDisparityMap (const std::string file_name,float* data_) {
    png::image< png::gray_pixel_16 > image(this->width,this->height);
    #pragma omp parallel
    {
     #pragma omp for
     for (int32_t v=0; v<this->height; v++) {
       for (int32_t u=0; u<this->width; u++) {
         if (isValid(data_,u,v))
          	image.set_pixel(u,v,(uint16_t)(std::max(getDisp(data_,u,v)*256.0,1.0)));
         else  
            image.set_pixel(u,v,0);
       }
     }
	}
    image.write(file_name);
  }

  	void setHeight(int height){
  		this->height=height;
  	}

  	void setWidth(int width){
  		this->width=width;
  	}

	int getHeight(void){
		return this->height;
	}
	int getWidth(void){
		return this->width;
	}
	int getIntensity(void){
		return this->intensity;
	}
	virtual ~imgio(){}
private:
	int height;
	int width;
	int intensity;
	float endianess;
	uchar color_type;
	uchar bit_depth;

};

#endif /* PGM_H_ */
