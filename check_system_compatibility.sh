#!/bin/bash

nvcomp=$(which nvcc)

if [ -f "$nvcomp" ]; 
then
	echo "Nvidia compiler found at path $nvcomp"
else
	echo "No nvidia compiler found in the system!"
	exit
fi

libpng=$(find / -name "libpng*.so" -print -quit )
if [ -f "$libpng" ]
then
	echo "libpng was detected under $libpng"
else
	echo "libpng was not found in the system!"
	exit
fi

pngpp=$(find / -name "png++*" -print -quit)
if [ -d "$pngpp" ]
then
	echo "png++ was detected under $pngpp"
else
	echo "png++ was not found in the system!"
	exit
fi

avx2=$(cat /proc/cpuinfo | grep -o -m 1 "avx2")
if [ "$avx2" == "avx2" ]
then
	echo "AVX2 support found" 
else
	echo "Your CPU does not support the AVX2 instruction set"
	exit
fi

echo "System check done. Please configure the makefiles appropriately."


