// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details


#pragma once

#include "StereoCommon.h"

/* Census Filters - basic versions */
void census9x7_mode8(uint8* source, uint64* dest, uint32 width, uint32 height);
void census9x7_mode8_16bit(uint16* source, uint64* dest, uint32 width, uint32 height);

/* Optimized Versions */
void census5x5_SSE(uint8* source, uint32* dest, uint32 width, uint32 height);
void census5x5_16bit_SSE(uint16* source, uint32* dest, uint32 width, uint32 height);

/* median */
void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height);

