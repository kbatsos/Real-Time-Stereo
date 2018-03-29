CPP=g++

CUSTOM_INC= ../../libs

SGM=../sgm

TARGET=census

all: $(TARGET)



# $(TARGET):$(TARGET).o #$(CUSTOM_INC)/pgm.o $(SGM)/FastFilters.o $(SGM)/StereoBMHelper.o
# 	$(CPP) -fopenmp -o $(TARGET) $(TARGET).o -lpng16 #$(CUSTOM_INC)/pgm.o $(SGM)/FastFilters.o $(SGM)/StereoBMHelper.o


$(TARGET):$(TARGET).cpp 
	$(CPP) -I${CUSTOM_INC} -I${SGM} -O3 -g3 -Wall -fmessage-length=0 -mavx2 -fopenmp -ftree-vectorize -fopt-info -std=c++11 -fPIC -MMD -MP  $(TARGET).cpp $(SGM)/FastFilters.cpp $(SGM)/StereoBMHelper.cpp  -o $(TARGET) -lpng16 

# $(CUSTOM_INC)pgm.o: $(CUSTOM_INC)/pgm.cpp 
# 	$(CPP) -c -03 -g3 -Wall -mavx2 -fopenmp -std=c++11 $(CUSTOM_INC)/pgm.cpp -o $(CUSTOM_INC)/pgm.o

# $(SGM)/FastFilters.o: $(SGM)/FastFilters.cpp
# 	$(CPP) -c -03 -g3 -Wall -mavx2 -fopenmp -std=c++11 $(SGM)/FastFilters.cpp -o $(SGM)/FastFilters.o	

# $(SGM)/StereoBMHelper.o: $(SGM)/StereoBMHelper.cpp
# 	$(CPP) -c -03 -g3 -Wall -mavx2 -fopenmp -std=c++11 $(SGM)/StereoBMHelper.cpp -o $(SGM)/StereoBMHelper.o	

clean:
	rm -rf $(TARGET) *.d *.o $(CUSTOM_INC)/*.o
