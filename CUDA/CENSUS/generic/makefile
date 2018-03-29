#Set the compiler
NV = nvcc

# location of the Boost Python include files and library

POST = ../../libs
PGM = ../../../libs
CUDA_LIBS = /usr/local/cuda-8.0/lib64/

# compile mesh classes
TARGET = census

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(NV) -lineinfo  -arch=sm_52 --restrict -Xptxas --allow-expensive-optimizations=true  -O3  -lpng16 -L$(CUDA_LIBS) -I$(PGM) -I$(POST) -Xcompiler -Wall -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -g3  $(TARGET).cu -o $(TARGET)
	
clean:
	 rm -rf $(TARGET) *.o $(PGM)/*.o
