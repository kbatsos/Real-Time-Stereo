# Real Time Stereo
Real Time Stereo code repository. For more information about the code please read the corresponding paper (soon available). 

The code includes CPU and GPU implementations of three basic block matching algorithms, CENSUS, NCC and SAD. For cost-optimization and post-processing,
on the CPU level, Spangenberg's [Large Scale Semi-Global Matching on the CPU](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6856419) is integrated, while on the GPU level we integrate
the pipeline proposed by Mei et. al [On Building an Accurate Stereo Matching System on Graphics Hardware](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6130280), as implemented
by Jure Zbontar [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://arxiv.org/abs/1510.05970).

# Compilation

The code has been tested on Ubuntu 16.04, with gcc 5.4 and cuda 8 and 9. We have included a simple image reader which supports pgm, pfm and png 
image formats. The only dependency of the code is png++ which comes with libpng. If you want to remove this dependency please modify imgio.hpp,
or use your own image reader/writer. Sample makefiles are provided. For the CPU versions, your CPU must support the AVX2 instruction set. If your CPU 
does not support this instructions set, please modify the -m flag according to the intel intrinsic instruction set your CPU supports. For NCC, code
that is explicitly written using AVX2 must be removed or replaced. If you like to contribute with an implementation using SSE 4.2 or lower please contact me.
