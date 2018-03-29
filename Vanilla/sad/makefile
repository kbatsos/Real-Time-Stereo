CPP=g++

CUSTOM_INC= ../../libs

TARGET=sad

all: $(TARGET)



$(TARGET):$(TARGET).o ../../libs/pgm.o
	$(CPP) -fopenmp -o $(TARGET) $(TARGET).o ../../libs/pgm.o -lpng16


$(TARGET).o:$(TARGET).cpp 
	$(CPP) -I${CUSTOM_INC} -c -O3 -g3 -Wall -fmessage-length=0 -mavx2 -fopenmp -ftree-vectorize -fopt-info -std=c++11 -fPIC -MMD -MP  $(TARGET).cpp -o $(TARGET).o 


../../libs/pgm.0: ../../libs/pgm.cpp
	$(CPP) -c -03 -g3 -Wall -mavx -fopenmp -std=c++11 ../../libs/pgm.cpp -o ../../libs/pgm.o

clean:
	rm -rf $(TARGET) *.o ../../libs/*.o
