CPP=g++

CUSTOM_INC= ../../libs

SGM=../sgm

TARGET=census

all: $(TARGET)


$(TARGET):$(TARGET).cpp 
	$(CPP) -I${CUSTOM_INC} -I${SGM} -O3 -g3 -Wall -fmessage-length=0 -mavx2 -fopenmp -ftree-vectorize -fopt-info -std=c++11 -fPIC -MMD -MP  $(TARGET).cpp $(SGM)/FastFilters.cpp $(SGM)/StereoBMHelper.cpp  -o $(TARGET) -lpng 

clean:
	rm -rf $(TARGET) *.d *.o $(CUSTOM_INC)/*.o
