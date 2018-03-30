# Real Time Stereo
Real Time Stereo code repository. For more information about the code please read the corresponding paper (soon available). 

The code includes CPU and GPU implementations of three basic block matching algorithms, CENSUS, NCC and SAD. For cost-optimization and post-processing,
on the CPU level, Spangenberg's [Large Scale Semi-Global Matching on the CPU](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6856419) is integrated, while on the GPU level we integrate
the pipeline proposed by Mei et. al [On Building an Accurate Stereo Matching System on Graphics Hardware](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6130280), as implemented
by Jure Zbontar [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://arxiv.org/abs/1510.05970).

# Compilation

The code has been tested on Ubuntu 16.04, with gcc 5.4 and cuda 8 and 9. 
A simple image reader/writer supporting pgm, pfm and png image formats. is included. The only dependency of the code is png++ which comes with libpng. If you want to remove this dependency please modify imgio.hpp, or use your own image reader/writer. 

Before compiling please run check_system_compatibility.sh.

`chmod +x check_system_compatibility.sh && ./check_system_compatibility.sh`

The script will search for the following components in your system:

- nvcc
- libpng
- png++
- AVX2

Feel free to skip this part if you know that your system supports these components.

## CPU Compilation

The CPU folder contains all elements needed for the CPU implementations. Navigate to the matcher you are interested and compile using `make`.

### For users with no AVX2 support

Navigate to folder CPU/sgm. In the file StereoBMHelper.cpp, comment out line 16: `#define USE_AVX2`. Then navigate to the folder of the matcher you are interested and modify the makefine from:  
`$(CPP) -I${CUSTOM_INC} -I${SGM} -O3 -g3 -Wall -fmessage-length=0 -mavx2 ....  `  
to  
`$(CPP) -I${CUSTOM_INC} -I${SGM} -O3 -g3 -Wall -fmessage-length=0 -mXXXX ....  `

where -mXXX should be replaced high the highest intel intrinsic intruction set your CPU supports. If you are unsure about what type of intrinsic instruction set your CPU supports run:

`cat /proc/cpuinfo`

In the flags section look for `sse# or avx#`.

For NCC, navigate to the CPU/ncc folder and open ncc.cpp. Comment out line 21: `#define USE_AVX2`.  


## GPU Compilation

Compilation on the GPU is rather straightforward. Navigate to the folder for the implementation you are interested in and type make. If your cuda libraries are located in a different directory than `/usr/local/cuda-#/lib64/` specify the directory in line 7. 


# How To Run

After compiling you can get a list of arguments you can pass to the executable either by providing no arguments, or using the `-h` option.


## Arguments
`-l` Left image or a file containing left image names  
`-r` Right image or a file containing right image names  
`-ndisp` Number of disparities  
`-wsize` One dimension of the window, assumes square windows.  
`-dopost` Defaults to false. If set activates sgm cost optimization  
`-list` By default the executable handles single files. Pass this argument if you use lists of image names  
`-out` Directory to save disparity images  
`-out_type` Output image type. Supports |pgm|pfm|png|disp(uint16 png format default)  
`-postconf` A configuration file containing cost-optimization and post-processing parameters  

For the fixed window implementations the `-wsize` is ignored. 

### Using lists of image names

In the root directory of the project we provide sample files, `leftimg.txt` and `rightimg.txt` containing lists of image names. The images *must* have the same size and the `-list` option *must* be provided.

### Optimization and post processing configuration

Sample files are provided under the CPU and CUDA folders. The files follow a key->value scheme. Please keep the keys names as is. 

#### CPU configuration parameters

`numStrips` the number of threads and stripes to use during cost optimization  
`InvalidDispCost` should be set to the maximum cost a matcher can produce  
`lrCheck` whether to perform left right check  
`MedianFilter` whether to perform median filtering  
`Paths` allowable values 2|4|8  
`subPixelRefine` whether to perform subpixel refinement through parabolic fit  
`NoPasses` Number of SGM passes  
`rlCheck` whether to perform right left check. (The right disparity map is not saved by default)  
`P1` sgm parameter  
`P2min` sgm parameter  
`Alpha` P2 modifier  
`Gamma` P2 modifier  

#### GPU configuration parameters

`pi1` SGM parameter  
`pi2` SGM parameter  
`tau_so` SGM grdient threshold to apply modifiers  
`alpha1` SGM pi2 modifier  
`sgm_q1` SGM pi2 modifier  
`sgm_q2` SGM pi2 modifier  
`alpha2` Bilateral filter parameter  
`sigma`  Bilateral filter kernel parameter  
`kernel_size` Median filter kernel size  









