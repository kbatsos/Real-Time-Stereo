// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details

#include "StereoCommon.h"
#include "FastFilters.h"

#include <smmintrin.h> // intrinsics
#include <emmintrin.h>
#include "string.h" // memset
#include "assert.h"

inline uint8* getPixel8(uint8* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

inline uint16* getPixel16(uint16* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

inline uint32* getPixel32(uint32* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

inline uint64* getPixel64(uint64* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

FORCEINLINE void testpixel(uint8* source, uint32 width, sint32 i, sint32 j,uint64& value, sint32 x, sint32 y)
{
    if (*getPixel8(source, width,j+x,i+y) - *getPixel8(source, width,j-x,i-y)>0) {
        value += 1;
    }
}

FORCEINLINE void testpixel2(uint8* source, uint32 width, sint32 i, sint32 j,uint64& value, sint32 x, sint32 y)
{
    value *= 2;
    uint8 result = *getPixel8(source, width,j+x,i+y) - *getPixel8(source, width,j-x,i-y)>0;
    value += result;
}

void census9x7_mode8(uint8* source, uint64* dest, uint32 width, uint32 height)
{
    memset(dest, 0, width*height*sizeof(uint64));

    // HWCS central symmetric with 3 central rows
    const int vert = 3;
    const int hor = 4;
    for (sint32 i=vert; i < (sint32)height-vert; i++) {
        for (sint32 j=hor; j < (sint32)width-hor; j++) {
            uint64 value = 0;
            testpixel(source, width, i,j, value,-4,-3);
            testpixel2(source, width, i,j, value,-3,-3);
            testpixel2(source, width, i,j, value,-2,-3);
            testpixel2(source, width, i,j, value,-1,-3);
            testpixel2(source, width, i,j, value,0,-3 );
            testpixel2(source, width, i,j, value,-4,-2);
            testpixel2(source, width, i,j, value,-3,-2);
            testpixel2(source, width, i,j, value,-2,-2);
            testpixel2(source, width, i,j, value,-1,-2);
            testpixel2(source, width, i,j, value,0,-2 );
            testpixel2(source, width, i,j, value,-4,-1);
            testpixel2(source, width, i,j, value,-3,-1);
            testpixel2(source, width, i,j, value,-2,-1);
            testpixel2(source, width, i,j, value,-1,-1);
            testpixel2(source, width, i,j, value,0,-1 );
            testpixel2(source, width, i,j, value,-4, 0);
            testpixel2(source, width, i,j, value,-3, 0);
            testpixel2(source, width, i,j, value,-2, 0);
            testpixel2(source, width, i,j, value,-1, 0);
            testpixel2(source, width, i,j, value,1,-3 );
            testpixel2(source, width, i,j, value,2,-3 );
            testpixel2(source, width, i,j, value,3,-3 );
            testpixel2(source, width, i,j, value,4,-3 );
            testpixel2(source, width, i,j, value,1,-2 );
            testpixel2(source, width, i,j, value,2,-2 );
            testpixel2(source, width, i,j, value,3,-2 );
            testpixel2(source, width, i,j, value,4,-2 );
            testpixel2(source, width, i,j, value,1,-1 );
            testpixel2(source, width, i,j, value,2,-1 );
            testpixel2(source, width, i,j, value,3,-1 );
            testpixel2(source, width, i,j, value,4,-1 );
            testpixel2(source, width, i,j, value,-4,-1);
            testpixel2(source, width, i,j, value,-3,-1);
            testpixel2(source, width, i,j, value,-2,-1);
            testpixel2(source, width, i,j, value,-1,-1);
            testpixel2(source, width, i,j, value,-4, 0);
            testpixel2(source, width, i,j, value,-3, 0);
            testpixel2(source, width, i,j, value,-2, 0);
            testpixel2(source, width, i,j, value,-1, 0);
            testpixel2(source, width, i,j, value,-4, 1);
            testpixel2(source, width, i,j, value,-3, 1);
            testpixel2(source, width, i,j, value,-2, 1);
            testpixel2(source, width, i,j, value,-1, 1);
            testpixel2(source, width, i,j, value,0,-1 );
            *getPixel64(dest,width,j,i) = (uint64)value;
        }
    }
}

FORCEINLINE void testpixel_16bit(uint16* source, uint32 width, sint32 i, sint32 j, uint64& value, sint32 x, sint32 y)
{
    if (*getPixel16(source, width, j + x, i + y) - *getPixel16(source, width, j - x, i - y) > 0) {
        value += 1;
    }
}

FORCEINLINE void testpixel2_16bit(uint16* source, uint32 width, sint32 i, sint32 j, uint64& value, sint32 x, sint32 y)
{
    value *= 2;
    uint8 result = *getPixel16(source, width, j + x, i + y) - *getPixel16(source, width, j - x, i - y) > 0;
    value += result;
}

void census9x7_mode8_16bit(uint16* source, uint64* dest, uint32 width, uint32 height)
{
    memset(dest, 0, width*height*sizeof(uint64));

    // HWCS central symmetric with 3 central rows
    const int vert = 3;
    const int hor = 4;
    for (sint32 i = vert; i < (sint32)height - vert; i++) {
        for (sint32 j = hor; j < (sint32)width - hor; j++) {
            uint64 value = 0;
            testpixel_16bit(source, width, i, j, value, -4, -3);
            testpixel2_16bit(source, width, i, j, value, -3, -3);
            testpixel2_16bit(source, width, i, j, value, -2, -3);
            testpixel2_16bit(source, width, i, j, value, -1, -3);
            testpixel2_16bit(source, width, i, j, value, 0, -3);
            testpixel2_16bit(source, width, i, j, value, -4, -2);
            testpixel2_16bit(source, width, i, j, value, -3, -2);
            testpixel2_16bit(source, width, i, j, value, -2, -2);
            testpixel2_16bit(source, width, i, j, value, -1, -2);
            testpixel2_16bit(source, width, i, j, value, 0, -2);
            testpixel2_16bit(source, width, i, j, value, -4, -1);
            testpixel2_16bit(source, width, i, j, value, -3, -1);
            testpixel2_16bit(source, width, i, j, value, -2, -1);
            testpixel2_16bit(source, width, i, j, value, -1, -1);
            testpixel2_16bit(source, width, i, j, value, 0, -1);
            testpixel2_16bit(source, width, i, j, value, -4, 0);
            testpixel2_16bit(source, width, i, j, value, -3, 0);
            testpixel2_16bit(source, width, i, j, value, -2, 0);
            testpixel2_16bit(source, width, i, j, value, -1, 0);
            testpixel2_16bit(source, width, i, j, value, 1, -3);
            testpixel2_16bit(source, width, i, j, value, 2, -3);
            testpixel2_16bit(source, width, i, j, value, 3, -3);
            testpixel2_16bit(source, width, i, j, value, 4, -3);
            testpixel2_16bit(source, width, i, j, value, 1, -2);
            testpixel2_16bit(source, width, i, j, value, 2, -2);
            testpixel2_16bit(source, width, i, j, value, 3, -2);
            testpixel2_16bit(source, width, i, j, value, 4, -2);
            testpixel2_16bit(source, width, i, j, value, 1, -1);
            testpixel2_16bit(source, width, i, j, value, 2, -1);
            testpixel2_16bit(source, width, i, j, value, 3, -1);
            testpixel2_16bit(source, width, i, j, value, 4, -1);
            testpixel2_16bit(source, width, i, j, value, -4, -1);
            testpixel2_16bit(source, width, i, j, value, -3, -1);
            testpixel2_16bit(source, width, i, j, value, -2, -1);
            testpixel2_16bit(source, width, i, j, value, -1, -1);
            testpixel2_16bit(source, width, i, j, value, -4, 0);
            testpixel2_16bit(source, width, i, j, value, -3, 0);
            testpixel2_16bit(source, width, i, j, value, -2, 0);
            testpixel2_16bit(source, width, i, j, value, -1, 0);
            testpixel2_16bit(source, width, i, j, value, -4, 1);
            testpixel2_16bit(source, width, i, j, value, -3, 1);
            testpixel2_16bit(source, width, i, j, value, -2, 1);
            testpixel2_16bit(source, width, i, j, value, -1, 1);
            testpixel2_16bit(source, width, i, j, value, 0, -1);
            *getPixel64(dest, width, j, i) = (uint64)value;
        }
    }
}


// census 5x5
// input uint8 image, output uint32 image
void census5x5_SSE(uint8* source, uint32* dest, uint32 width, uint32 height)
{
    uint32* dst = dest;
    const uint8* src = source;

    // input lines 0,1,2
    const uint8* i0 = src;

    // output at first result
    uint32* result = dst + 2*width;
    const uint8* const end_input = src + width*height;

    /* expand mask */
    __m128i expandByte1_First4 =  _mm_set_epi8(0x80u, 0x80u, 0x80u, 0x03u,0x80u, 0x80u, 0x80u, 0x02u,
                                               0x80u, 0x80u, 0x80u, 0x01u,0x80u, 0x80u, 0x80u, 0x00u);

    __m128i expandByte2_First4 = _mm_set_epi8(0x80u, 0x80u, 0x03u, 0x80u,0x80u, 0x80u, 0x02u, 0x80u,
                                              0x80u, 0x80u, 0x01u, 0x80u,0x80u, 0x80u, 0x00u, 0x80u);

    __m128i expandByte3_First4  = _mm_set_epi8(0x80u, 0x03u, 0x80u, 0x80u,0x80u, 0x02u, 0x80u, 0x80u,
                                               0x80u, 0x01u, 0x80u, 0x80u,0x80u, 0x00u, 0x80u, 0x80u);

    /* xor with 0x80, as it is a signed compare */
    __m128i l2_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+2*width )) ,_mm_set1_epi8('\x80'));
    __m128i l3_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+3*width )) ,_mm_set1_epi8('\x80'));
    __m128i l4_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+4*width )) ,_mm_set1_epi8('\x80'));  
    __m128i l1_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+width )) ,_mm_set1_epi8('\x80'));
    __m128i l0_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0 )) ,_mm_set1_epi8('\x80'));

    i0 += 16;

    __m128i lastResult = _mm_setzero_si128();

    for( ; i0+4*width < end_input; i0 += 16 ) {

            /* parallel 16 pixel processing */
            const __m128i l0_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0 )) ,_mm_set1_epi8('\x80'));
            const __m128i l1_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+width )) ,_mm_set1_epi8('\x80'));
            const __m128i l2_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+2*width )) ,_mm_set1_epi8('\x80'));
            const __m128i l3_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+3*width )) ,_mm_set1_epi8('\x80'));
            const __m128i l4_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+4*width )) ,_mm_set1_epi8('\x80'));

            /*  0  1  2  3  4
                5  6  7  8  9
               10 11  c 12 13
               14 15 16 17 18
               19 20 21 22 23 */
            /* r/h is result, v is pixelvalue */

            /* pixel c */
            __m128i pixelcv = _mm_alignr_epi8(l2_register_next,l2_register, 2);

            /* pixel 0*/ 
            __m128i pixel0h = _mm_cmplt_epi8(l0_register,pixelcv);
            
            /* pixel 1*/ 
            __m128i pixel1v = _mm_alignr_epi8(l0_register_next,l0_register, 1);
            __m128i pixel1h = _mm_cmplt_epi8(pixel1v,pixelcv);

            /* pixel 2 */
            __m128i pixel2v = _mm_alignr_epi8(l0_register_next,l0_register, 2);
            __m128i pixel2h = _mm_cmplt_epi8(pixel2v,pixelcv);

            /* pixel 3 */
            __m128i pixel3v = _mm_alignr_epi8(l0_register_next,l0_register, 3);
            __m128i pixel3h = _mm_cmplt_epi8(pixel3v,pixelcv);

            /* pixel 4 */
            __m128i pixel4v = _mm_alignr_epi8(l0_register_next,l0_register, 4);
            __m128i pixel4h = _mm_cmplt_epi8(pixel4v,pixelcv);

            /** line  **/
            /* pixel 5 */
            __m128i pixel5h = _mm_cmplt_epi8(l1_register,pixelcv);

            /* pixel 6 */
            __m128i pixel6v = _mm_alignr_epi8(l1_register_next,l1_register, 1);
            __m128i pixel6h = _mm_cmplt_epi8(pixel6v,pixelcv);

            /* pixel 7 */
            __m128i pixel7v = _mm_alignr_epi8(l1_register_next,l1_register, 2);
            __m128i pixel7h = _mm_cmplt_epi8(pixel7v,pixelcv);

            /* pixel 8 */
            __m128i pixel8v = _mm_alignr_epi8(l1_register_next,l1_register, 3);
            __m128i pixel8h = _mm_cmplt_epi8(pixel8v,pixelcv);

            /* pixel 9 */
            __m128i pixel9v = _mm_alignr_epi8(l1_register_next,l1_register, 4);
            __m128i pixel9h = _mm_cmplt_epi8(pixel9v,pixelcv);

            /* create bitfield part 1*/
            __m128i resultByte1 = _mm_and_si128(_mm_set1_epi8(128u),pixel0h);
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(64),pixel1h));
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(32),pixel2h));
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(16),pixel3h));
            __m128i resultByte1b = _mm_and_si128(_mm_set1_epi8(8),pixel4h);
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(4),pixel5h));
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(2),pixel6h));
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(1),pixel7h));
            resultByte1 = _mm_or_si128(resultByte1, resultByte1b);

            /** line **/
            /* pixel 10 */
            __m128i pixel10h = _mm_cmplt_epi8(l2_register,pixelcv);

            /* pixel 11 */
            __m128i pixel11v = _mm_alignr_epi8(l2_register_next,l2_register, 1);
            __m128i pixel11h = _mm_cmplt_epi8(pixel11v,pixelcv);

            /* pixel 12 */
            __m128i pixel12v = _mm_alignr_epi8(l2_register_next,l2_register, 3);
            __m128i pixel12h = _mm_cmplt_epi8(pixel12v,pixelcv);
            
            /* pixel 13 */
            __m128i pixel13v = _mm_alignr_epi8(l2_register_next,l2_register, 4);
            __m128i pixel13h = _mm_cmplt_epi8(pixel13v,pixelcv);

            /* line */
            /* pixel 14 */
            __m128i pixel14h = _mm_cmplt_epi8(l3_register,pixelcv);

            /* pixel 15 */
            __m128i pixel15v = _mm_alignr_epi8(l3_register_next,l3_register, 1);
            __m128i pixel15h = _mm_cmplt_epi8(pixel15v,pixelcv);

            /* pixel 16 */
            __m128i pixel16v = _mm_alignr_epi8(l3_register_next,l3_register, 2);
            __m128i pixel16h = _mm_cmplt_epi8(pixel16v,pixelcv);

            /* pixel 17 */
            __m128i pixel17v = _mm_alignr_epi8(l3_register_next,l3_register, 3);
            __m128i pixel17h = _mm_cmplt_epi8(pixel17v,pixelcv);

            /* pixel 18 */
            __m128i pixel18v = _mm_alignr_epi8(l3_register_next,l3_register, 4);
            __m128i pixel18h = _mm_cmplt_epi8(pixel18v,pixelcv);

            /* create bitfield part 2 */
            __m128i resultByte2 = _mm_and_si128(_mm_set1_epi8(128u),pixel8h);
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(64),pixel9h));
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(32),pixel10h));
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(16),pixel11h));
            __m128i resultByte2b = _mm_and_si128(_mm_set1_epi8(8),pixel12h);
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(4),pixel13h));
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(2),pixel14h));
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(1),pixel15h));
            resultByte2 = _mm_or_si128(resultByte2, resultByte2b);

            /* line */
            /* pixel 19 */
            __m128i pixel19h = _mm_cmplt_epi8(l4_register,pixelcv);

            /* pixel 20 */
            __m128i pixel20v = _mm_alignr_epi8(l4_register_next,l4_register, 1);
            __m128i pixel20h = _mm_cmplt_epi8(pixel20v,pixelcv);

            /* pixel 21 */
            __m128i pixel21v = _mm_alignr_epi8(l4_register_next,l4_register, 2);
            __m128i pixel21h = _mm_cmplt_epi8(pixel21v,pixelcv);

            /* pixel 22 */
            __m128i pixel22v = _mm_alignr_epi8(l4_register_next,l4_register, 3);
            __m128i pixel22h = _mm_cmplt_epi8(pixel22v,pixelcv);

            /* pixel 23 */
            __m128i pixel23v = _mm_alignr_epi8(l4_register_next,l4_register, 4);
            __m128i pixel23h = _mm_cmplt_epi8(pixel23v,pixelcv);

            /* create bitfield part 3*/

            __m128i resultByte3 = _mm_and_si128(_mm_set1_epi8(128u),pixel16h);
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(64),pixel17h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(32),pixel18h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(16),pixel19h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(8),pixel20h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(4),pixel21h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(2),pixel22h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(1),pixel23h));

            /* blend the first part together */
            __m128i resultPart1_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
            __m128i resultPart1_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
            __m128i resultPart1 = _mm_or_si128(resultPart1_1,resultPart1_2);
            __m128i resultPart1_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
            resultPart1 = _mm_or_si128(resultPart1,resultPart1_3);
            
            /* rotate result bytes */
            /* replace by _mm_alignr_epi8 */
            resultByte1 = _mm_shuffle_epi32(resultByte1, _MM_SHUFFLE(0,3,2,1));
            resultByte2 = _mm_shuffle_epi32(resultByte2, _MM_SHUFFLE(0,3,2,1));
            resultByte3 = _mm_shuffle_epi32(resultByte3, _MM_SHUFFLE(0,3,2,1));

            /* blend the second part together */
            __m128i resultPart2_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
            __m128i resultPart2_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
            __m128i resultPart2 = _mm_or_si128(resultPart2_1,resultPart2_2);
            __m128i resultPart2_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
            resultPart2 = _mm_or_si128(resultPart2,resultPart2_3);

            /* rotate result bytes */
            resultByte1 = _mm_shuffle_epi32(resultByte1, _MM_SHUFFLE(0,3,2,1));
            resultByte2 = _mm_shuffle_epi32(resultByte2, _MM_SHUFFLE(0,3,2,1));
            resultByte3 = _mm_shuffle_epi32(resultByte3, _MM_SHUFFLE(0,3,2,1));

            /* blend the third part together */
            __m128i resultPart3_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
            __m128i resultPart3_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
            __m128i resultPart3 = _mm_or_si128(resultPart3_1,resultPart3_2);
            __m128i resultPart3_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
            resultPart3 = _mm_or_si128(resultPart3,resultPart3_3);

            /* rotate result bytes */
            resultByte1 = _mm_shuffle_epi32(resultByte1, _MM_SHUFFLE(0,3,2,1));
            resultByte2 = _mm_shuffle_epi32(resultByte2, _MM_SHUFFLE(0,3,2,1));
            resultByte3 = _mm_shuffle_epi32(resultByte3, _MM_SHUFFLE(0,3,2,1));

            /* blend the fourth part together */
            __m128i resultPart4_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
            __m128i resultPart4_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
            __m128i resultPart4 = _mm_or_si128(resultPart4_1,resultPart4_2);
            __m128i resultPart4_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
            resultPart4 = _mm_or_si128(resultPart4,resultPart4_3);

            /* shift because of offsets */
            _mm_store_si128((__m128i*)result, _mm_alignr_epi8(resultPart1, lastResult, 8));
            _mm_store_si128((__m128i*)(result+4), _mm_alignr_epi8(resultPart2, resultPart1, 8) );
            _mm_store_si128((__m128i*)(result+8), _mm_alignr_epi8(resultPart3, resultPart2, 8) ); 
            _mm_store_si128((__m128i*)(result+12), _mm_alignr_epi8(resultPart4, resultPart3, 8) ); 

            result += 16;
            lastResult = resultPart4;

            /*load next */
            l0_register = l0_register_next;
            l1_register = l1_register_next;
            l2_register = l2_register_next;
            l3_register = l3_register_next;
            l4_register = l4_register_next;

    }
    /* last pixels */
    {
        int i = height - 3;
        for (sint32 j=width-16+2; j < (sint32)width-2; j++) {
            const int centerValue = *getPixel8(source, width,j,i);
            uint32 value = 0;
            for (sint32 x=-2; x <= 2; x++) {
                for (sint32 y=-2; y <= 2; y++) {
                    if (x!=0 || y!=0) {
                        value *= 2;
                        if (centerValue >  *getPixel8(source, width,j+y,i+x)) {
                            value += 1;
                        }
                    }
                }     
            } 
            *getPixel32(dest,width,j,i) = value;
        }
    }
}

// census 5x5
// input uint16 image, output uint32 image
// 2.56ms/2, 768x480, 9.2 cycles/pixel
void census5x5_16bit_SSE(uint16* source, uint32* dest, uint32 width, uint32 height)
{
    uint32* dst = dest;
    const uint16* src = source;
    
    // memsets just for the upper and lower two lines, not really necessary
    memset(dest, 0, width*2*sizeof(uint32));
    memset(dest+width*(height-2), 0, width*2*sizeof(uint32));

    // input lines 0,1,2
    const uint16* i0 = src;
    const uint16* i1 = src+width;
    const uint16* i2 = src+2*width;
    const uint16* i3 = src+3*width;
    const uint16* i4 = src+4*width;

    // output at first result
    uint32* result = dst + 2*width;
    const uint16* const end_input = src + width*height;

    /* expand mask */
    __m128i expandLowerMask  = _mm_set_epi8(0x06, 0x06, 0x06, 0x06, 0x04, 0x04, 0x04, 0x04,
                                            0x02, 0x02, 0x02, 0x02, 0x00, 0x00, 0x00, 0x00);

    __m128i expandUpperMask = _mm_set_epi8(0x0E, 0x0E, 0x0E, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C,
                                           0x0A, 0x0A, 0x0A, 0x0A, 0x08, 0x08, 0x08, 0x08);

    __m128i blendB1B2Mask = _mm_set_epi8(0x00u, 0x00u, 0x80u, 0x00u, 0x00u, 0x00u, 0x80u, 0x00u,
                                         0x00u, 0x00u, 0x80u, 0x00u, 0x00u, 0x00u, 0x80u, 0x00u);

    __m128i blendB1B2B3Mask  = _mm_set_epi8(0x00u, 0x80u, 0x00u, 0x00u, 0x00u, 0x80u, 0x00u, 0x00u,
                                            0x00u, 0x80u, 0x00u, 0x00u, 0x00u, 0x80u, 0x00u, 0x00u);

    __m128i l2_register = _mm_stream_load_si128( (__m128i*)( i2 ) );
    __m128i l3_register = _mm_stream_load_si128( (__m128i*)( i3 ) );
    __m128i l4_register = _mm_stream_load_si128( (__m128i*)( i4 ) );  
    __m128i l1_register = _mm_stream_load_si128( (__m128i*)( i1 ) );
    __m128i l0_register = _mm_stream_load_si128( (__m128i*)( i0 ) );

    i0 += 8;
    i1 += 8;
    i2 += 8;
    i3 += 8;
    i4 += 8;

    __m128i lastResultLower = _mm_setzero_si128();

    for( ; i4 < end_input; i0 += 8, i1 += 8, i2 += 8, i3+=8, i4+=8 ) {

            /* parallel 16 pixel processing */
            const __m128i l0_register_next = _mm_stream_load_si128( (__m128i*)( i0 ) );
            const __m128i l1_register_next = _mm_stream_load_si128( (__m128i*)( i1 ) );
            const __m128i l2_register_next = _mm_stream_load_si128( (__m128i*)( i2 ) );
            const __m128i l3_register_next = _mm_stream_load_si128( (__m128i*)( i3 ) );
            const __m128i l4_register_next = _mm_stream_load_si128( (__m128i*)( i4 ) );

            /*  0  1  2  3  4
                5  6  7  8  9
               10 11  c 12 13
               14 15 16 17 18
               19 20 21 22 23 */
            /* r/h is result, v is pixelvalue */

            /* pixel c */
            __m128i pixelcv = _mm_alignr_epi8(l2_register_next,l2_register, 4);

            /* pixel 0*/ 
            __m128i pixel0h = _mm_cmplt_epi16(l0_register,pixelcv);
            
            /* pixel 1*/ 
            __m128i pixel1v = _mm_alignr_epi8(l0_register_next,l0_register, 2);
            __m128i pixel1h = _mm_cmplt_epi16(pixel1v,pixelcv);

            /* pixel 2 */
            __m128i pixel2v = _mm_alignr_epi8(l0_register_next,l0_register, 4);
            __m128i pixel2h = _mm_cmplt_epi16(pixel2v,pixelcv);

            /* pixel 3 */
            __m128i pixel3v = _mm_alignr_epi8(l0_register_next,l0_register, 6);
            __m128i pixel3h = _mm_cmplt_epi16(pixel3v,pixelcv);

            /* pixel 4 */
            __m128i pixel4v = _mm_alignr_epi8(l0_register_next,l0_register, 8);
            __m128i pixel4h = _mm_cmplt_epi16(pixel4v,pixelcv);

            /** line  **/
            /* pixel 5 */
            __m128i pixel5h = _mm_cmplt_epi16(l1_register,pixelcv);

            /* pixel 6 */
            __m128i pixel6v = _mm_alignr_epi8(l1_register_next,l1_register, 2);
            __m128i pixel6h = _mm_cmplt_epi16(pixel6v,pixelcv);

            /* pixel 7 */
            __m128i pixel7v = _mm_alignr_epi8(l1_register_next,l1_register, 4);
            __m128i pixel7h = _mm_cmplt_epi16(pixel7v,pixelcv);

            /* pixel 8 */
            __m128i pixel8v = _mm_alignr_epi8(l1_register_next,l1_register, 6);
            __m128i pixel8h = _mm_cmplt_epi16(pixel8v,pixelcv);

            /* pixel 9 */
            __m128i pixel9v = _mm_alignr_epi8(l1_register_next,l1_register, 8);
            __m128i pixel9h = _mm_cmplt_epi16(pixel9v,pixelcv);

            /* create bitfield part 1*/
            __m128i resultByte1 = _mm_and_si128(_mm_set1_epi8(128u),pixel0h);
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(64),pixel1h));
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(32),pixel2h));
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(16),pixel3h));
            __m128i resultByte1b = _mm_and_si128(_mm_set1_epi8(8),pixel4h);
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(4),pixel5h));
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(2),pixel6h));
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(1),pixel7h));
            resultByte1 = _mm_or_si128(resultByte1, resultByte1b);

            /** line **/
            /* pixel 10 */
            __m128i pixel10h = _mm_cmplt_epi16(l2_register,pixelcv);

            /* pixel 11 */
            __m128i pixel11v = _mm_alignr_epi8(l2_register_next,l2_register, 2);
            __m128i pixel11h = _mm_cmplt_epi16(pixel11v,pixelcv);

            /* pixel 12 */
            __m128i pixel12v = _mm_alignr_epi8(l2_register_next,l2_register, 6);
            __m128i pixel12h = _mm_cmplt_epi16(pixel12v,pixelcv);
            
            /* pixel 13 */
            __m128i pixel13v = _mm_alignr_epi8(l2_register_next,l2_register, 8);
            __m128i pixel13h = _mm_cmplt_epi16(pixel13v,pixelcv);

            /* line */
            /* pixel 14 */
            __m128i pixel14h = _mm_cmplt_epi16(l3_register,pixelcv);

            /* pixel 15 */
            __m128i pixel15v = _mm_alignr_epi8(l3_register_next,l3_register, 2);
            __m128i pixel15h = _mm_cmplt_epi16(pixel15v,pixelcv);

            /* pixel 16 */
            __m128i pixel16v = _mm_alignr_epi8(l3_register_next,l3_register, 4);
            __m128i pixel16h = _mm_cmplt_epi16(pixel16v,pixelcv);

            /* pixel 17 */
            __m128i pixel17v = _mm_alignr_epi8(l3_register_next,l3_register, 6);
            __m128i pixel17h = _mm_cmplt_epi16(pixel17v,pixelcv);

            /* pixel 18 */
            __m128i pixel18v = _mm_alignr_epi8(l3_register_next,l3_register, 8);
            __m128i pixel18h = _mm_cmplt_epi16(pixel18v,pixelcv);

            /* create bitfield part 2 */
            __m128i resultByte2 = _mm_and_si128(_mm_set1_epi8(128u),pixel8h);
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(64),pixel9h));
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(32),pixel10h));
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(16),pixel11h));
            __m128i resultByte2b = _mm_and_si128(_mm_set1_epi8(8),pixel12h);
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(4),pixel13h));
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(2),pixel14h));
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(1),pixel15h));
            resultByte2 = _mm_or_si128(resultByte2, resultByte2b);

            /* line */
            /* pixel 19 */
            __m128i pixel19h = _mm_cmplt_epi16(l4_register,pixelcv);

            /* pixel 20 */
            __m128i pixel20v = _mm_alignr_epi8(l4_register_next,l4_register, 2);
            __m128i pixel20h = _mm_cmplt_epi16(pixel20v,pixelcv);

            /* pixel 21 */
            __m128i pixel21v = _mm_alignr_epi8(l4_register_next,l4_register, 4);
            __m128i pixel21h = _mm_cmplt_epi16(pixel21v,pixelcv);

            /* pixel 22 */
            __m128i pixel22v = _mm_alignr_epi8(l4_register_next,l4_register, 6);
            __m128i pixel22h = _mm_cmplt_epi16(pixel22v,pixelcv);

            /* pixel 23 */
            __m128i pixel23v = _mm_alignr_epi8(l4_register_next,l4_register, 8);
            __m128i pixel23h = _mm_cmplt_epi16(pixel23v,pixelcv);

            /* create bitfield part 3*/

            __m128i resultByte3 = _mm_and_si128(_mm_set1_epi8(128u),pixel16h);
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(64),pixel17h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(32),pixel18h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(16),pixel19h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(8),pixel20h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(4),pixel21h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(2),pixel22h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(1),pixel23h));

            /* blend byte 1 and byte 2,then byte3, lower part */
            __m128i resultByte1Lower = _mm_shuffle_epi8(resultByte1, expandLowerMask);
            __m128i resultByte2Lower = _mm_shuffle_epi8(resultByte2, expandLowerMask);
            __m128i blendB1B2 = _mm_blendv_epi8(resultByte1Lower,resultByte2Lower,blendB1B2Mask);
            blendB1B2 = _mm_and_si128(blendB1B2, _mm_set1_epi32(0x00FFFFFF)); // zero first byte
            __m128i blendB1B2B3L = _mm_blendv_epi8(blendB1B2,_mm_shuffle_epi8(resultByte3, expandLowerMask),blendB1B2B3Mask);

            /* blend byte 1 and byte 2,then byte3, upper part */
            __m128i resultByte1Upper = _mm_shuffle_epi8(resultByte1, expandUpperMask);
            __m128i resultByte2Upper = _mm_shuffle_epi8(resultByte2, expandUpperMask);
            blendB1B2 = _mm_blendv_epi8(resultByte1Upper,resultByte2Upper,blendB1B2Mask);
            blendB1B2 = _mm_and_si128(blendB1B2, _mm_set1_epi32(0x00FFFFFF)); // zero first byte
            __m128i blendB1B2B3H = _mm_blendv_epi8(blendB1B2,_mm_shuffle_epi8(resultByte3, expandUpperMask),blendB1B2B3Mask);

            /* shift because of offsets */
            __m128i c = _mm_alignr_epi8(blendB1B2B3L, lastResultLower, 8);
            _mm_store_si128((__m128i*)result, c);
            _mm_store_si128((__m128i*)(result+4), _mm_alignr_epi8(blendB1B2B3H, blendB1B2B3L, 8) ); 

            result += 8;
            lastResultLower = blendB1B2B3H;

            /*load next */
            l0_register = l0_register_next;
            l1_register = l1_register_next;
            l2_register = l2_register_next;
            l3_register = l3_register_next;
            l4_register = l4_register_next;

    }
    /* last 8 pixels */
    {
        int i = height - 3;
        for (sint32 j=width-8; j < (sint32)width-2; j++) {
            const int centerValue = *getPixel16(source, width,j,i);
            uint32 value = 0;
            for (sint32 x=-2; x <= 2; x++) {
                for (sint32 y=-2; y <= 2; y++) {
                    if (x!=0 || y!=0) {
                        value *= 2;
                        if (centerValue >  *getPixel16(source, width,j+y,i+x)) {
                            value += 1;
                        }
                    }
                }     
            } 
            *getPixel32(dest,width,j,i) = value;
        }
        *getPixel32(dest,width,width-2,i) = 255;
        *getPixel32(dest,width,width-1,i) = 255;
    }
}

inline void vecSortandSwap(__m128& a, __m128& b)
{
    __m128 temp = a;
    a = _mm_min_ps(a,b);
    b = _mm_max_ps(temp,b);
}

void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height)
{
    // check width restriction
    assert(width % 4 == 0);
    
    float32* destStart = dest;
    //  lines
    float32* line1 = source;
    float32* line2 = source + width;
    float32* line3 = source + 2*width;

    float32* end = source + width*height;

    dest += width;
    __m128 lastMedian = _mm_setzero_ps();

    do {
        // fill value
        const __m128 l1_reg = _mm_load_ps(line1);
        const __m128 l1_reg_next = _mm_load_ps(line1+4);
        __m128 v0 = l1_reg;
        __m128 v1 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next),_mm_castps_si128(l1_reg), 4));
        __m128 v2 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next),_mm_castps_si128(l1_reg), 8));

        const __m128 l2_reg = _mm_load_ps(line2);
        const __m128 l2_reg_next = _mm_load_ps(line2+4);
        __m128 v3 = l2_reg;
        __m128 v4 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next),_mm_castps_si128(l2_reg), 4));
        __m128 v5 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next),_mm_castps_si128(l2_reg), 8));

        const __m128 l3_reg = _mm_load_ps(line3);
        const __m128 l3_reg_next = _mm_load_ps(line3+4);
        __m128 v6 = l3_reg;
        __m128 v7 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next),_mm_castps_si128(l3_reg), 4));
        __m128 v8 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next),_mm_castps_si128(l3_reg), 8));

        // find median through sorting network
        vecSortandSwap(v1, v2) ; vecSortandSwap(v4, v5) ; vecSortandSwap(v7, v8) ;
        vecSortandSwap(v0, v1) ; vecSortandSwap(v3, v4) ; vecSortandSwap(v6, v7) ;
        vecSortandSwap(v1, v2) ; vecSortandSwap(v4, v5) ; vecSortandSwap(v7, v8) ;
        vecSortandSwap(v0, v3) ; vecSortandSwap(v5, v8) ; vecSortandSwap(v4, v7) ;
        vecSortandSwap(v3, v6) ; vecSortandSwap(v1, v4) ; vecSortandSwap(v2, v5) ;
        vecSortandSwap(v4, v7) ; vecSortandSwap(v4, v2) ; vecSortandSwap(v6, v4) ;
        vecSortandSwap(v4, v2) ; 

        // comply to alignment restrictions
        const __m128i c = _mm_alignr_epi8(_mm_castps_si128(v4), _mm_castps_si128(lastMedian), 12);
        _mm_store_si128((__m128i*)dest, c);
        lastMedian = v4;

        dest+=4; line1+=4; line2+=4; line3+=4;

    } while (line3+4+4 <= end);

    memcpy(destStart, source, sizeof(float32)*(width+1));
    memcpy(destStart+width*height-width-1-3, source+width*height-width-1-3, sizeof(float32)*(width+1+3));
}

