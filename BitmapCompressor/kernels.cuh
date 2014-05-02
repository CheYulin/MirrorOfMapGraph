/**
 Copyright 2013-2014 SYSTAP, LLC.  http://www.systap.com

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 This work was (partially) funded by the DARPA XDATA program under
 AFRL Contract #FA8750-13-C-0002.

 This material is based upon work supported by the Defense Advanced
 Research Projects Agency (DARPA) under Contract No. D14PC00029.
 */

/*
 * kernels.cuh
 *
 *  Created on: Apr 29, 2014
 *      Author: zhisong
 */

#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include <assert.h>

__global__ void generateExtendedBitmap(unsigned int bitmap_size, unsigned char* bitmap, unsigned int* bitmap_extended)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int byte_size = (bitmap_size + 8 - 1) / 8;
  int start = tid * 31 / 8;
  int end = (tid * 31 + 30) / 8;
  unsigned int tmp_extended = 0;
  int start_offset;
  unsigned int b;

  if (start < byte_size)
  {
    start_offset = (tid * 31) % 8;

    b = (unsigned int)bitmap[start];
//    if (tid == 0)
//      printf("blockIdx.x=%d, threadIdx.x=%d, bitmap_size=%d, start=%d, end=%d, start_offset=%d, b=%d\n",
//             blockIdx.x, threadIdx.x, bitmap_size, start, end, start_offset, b);
    b <<= (3 * 8 + start_offset);
//    if (tid == 0)
//      printf("blockIdx.x=%d, threadIdx.x=%d, start=%d, end=%d, start_offset=%d, b=%d\n",
//             blockIdx.x, threadIdx.x, start, end, start_offset, b);
    tmp_extended += b;
//    if (tid == 0)
//      printf("blockIdx.x=%d, threadIdx.x=%d, start=%d, end=%d, start_offset=%d, bitmap_extended[tid]=%d\n",
//             blockIdx.x, threadIdx.x, start, end, start_offset, tmp_extended);
  }

  unsigned int bit_count = 8 - start_offset;
  for (int i = start + 1; i < end; i++)
  {

    if (i < byte_size)
    {
      b = (unsigned int)bitmap[i];
//      if (tid == 0)
//        printf("blockIdx.x=%d, threadIdx.x=%d, i=%d, start=%d, end=%d, start_offset=%d, bit_count=%d, b=%d\n",
//               blockIdx.x, threadIdx.x, i, start, end, start_offset, bit_count, b);
      b <<= 3 * 8 - bit_count;
//      if (tid == 0)
//        printf("blockIdx.x=%d, threadIdx.x=%d, i=%d, start=%d, end=%d, start_offset=%d, bit_count=%d, b=%d\n",
//               blockIdx.x, threadIdx.x, i, start, end, start_offset, bit_count, b);
      tmp_extended += b;
//      if (tid == 0)
//        printf("blockIdx.x=%d, threadIdx.x=%d, i=%d, start=%d, end=%d, start_offset=%d, bit_count=%d, bitmap_extended[tid]=%d\n",
//               blockIdx.x, threadIdx.x, i, start, end, start_offset, bit_count, tmp_extended);
      bit_count += 8;
    }

  }

  if (end < byte_size)
  {

    int end_offset = (tid * 31 + 30) % 8;
    if (end_offset != 31 - bit_count - 1)
    {
      printf("Error: blockIdx.x=%d, threadIdx.x=%d, end_offset=%d, bit_count=%d\n", blockIdx.x, threadIdx.x, end_offset, bit_count);
    }
    b = (unsigned int)bitmap[end];
    //    if (tid == 0)
    //      printf("blockIdx.x=%d, threadIdx.x=%d, start=%d, end=%d, end_offset=%d, bit_count=%d, b=%d\n",
    //             blockIdx.x, threadIdx.x, start, end, end_offset, bit_count, b);
    b >>= 8 - end_offset - 1;
    //    if (tid == 0)
    //      printf("blockIdx.x=%d, threadIdx.x=%d, start=%d, end=%d, end_offset=%d, bit_count=%d, b=%d\n",
    //             blockIdx.x, threadIdx.x, start, end, end_offset, bit_count, b);
    b <<= 1;
    //    if (tid == 0)
    //      printf("blockIdx.x=%d, threadIdx.x=%d, start=%d, end=%d, end_offset=%d, bit_count=%d, b=%d\n",
    //             blockIdx.x, threadIdx.x, start, end, end_offset, bit_count, b);
    tmp_extended += b;
    //    if (tid == 0)
    //      printf("blockIdx.x=%d, threadIdx.x=%d, start=%d, end=%d, start_offset=%d, bit_count=%d, bitmap_extended[tid]=%d\n",
    //             blockIdx.x, threadIdx.x, start, end, start_offset, bit_count, tmp_extended);
  }

  if (tmp_extended == 0 || tmp_extended == 4294967294) //bitmap_extended == 0 || bitmap_extended== 111...111110
    tmp_extended += 1;

  bitmap_extended[tid] = tmp_extended;

}

__inline__ __device__ bool bit_equal(const unsigned int& A, const unsigned int& B, const int &left, const int &right)
{
  for (int i = left; i < right; i++)
  {
    unsigned int mask = 1 << (31 - i);
    if ((A & mask) != (B & mask))
      return false;
  }
  return true;
}

__inline__ __device__ int count_ones(const unsigned int& A, const int &left, const int &right)
{
  int count = 0;
  for (int i = left; i < right; i++)
  {
    unsigned int mask = 1 << (31 - i);
    if (A & mask)
      count++;
  }
  return count;
}

__inline__ __device__ int position_of_one(const unsigned int& A)
{

  for (int pos = 0; pos < 32; pos++)
  {
    unsigned int mask = 1 << (31 - pos);
    if (A & mask)
      return pos; //return the pos of the first found one
  }
  return -1; //no ones in A
}

__device__ bool NextNotEqual(const unsigned int& A, const unsigned int& B)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_ones = count_ones(B, 0, 31);
  //	if(tid==1) 
  //  printf("tid=%d, A==1: %d, count_ones: %d\n", tid, A == 1, num_ones);
  ////	if(tid==1) 
  //  printf("tid=%d, A == 4294967295: %d, count_ones: %d\n", tid, A == 4294967295, num_ones);
  ////	if(tid==1) 
  //  printf("tid=%d, bit_equal(A, B, 30, 32): %d, (A & 1) == 0: %d\n", tid, bit_equal(A, B, 30, 32), (A & 1) == 0);
  if (A == 1 && num_ones == 1)
  {
    return false;
  }
  else if (A == 4294967295 && num_ones == 30)
  {
    return false;
  }
  else
  {
    return (!bit_equal(A, B, 30, 32) || (A & 1) == 0);
  }
}

__global__ void initF(unsigned int word_size, unsigned int* bitmap_extended, unsigned int* bitmap_F)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < word_size - 1)
  {
    if (NextNotEqual(bitmap_extended[tid], bitmap_extended[tid + 1]))
      bitmap_F[tid] = 1;
    else
      bitmap_F[tid] = 0;
  }
  if (tid == 0)
  {
    bitmap_F[word_size - 1] = 1;
  }
}

__global__ void initT1(unsigned int word_size, unsigned int* F, unsigned int* SF, unsigned int* T1)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < word_size)
  {
    if (F[tid] == 1)
      T1[SF[tid]] = tid + 1;
  }
}

__global__ void initT2(unsigned int m, unsigned int* T1, unsigned int* T2)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < m)
  {
    int count = T1[tid];
    if (tid != 0)
      count -= T1[tid - 1];

    T2[tid] = ceil((double)count / ((1 << 25) - 1));
  }
}

__inline__ __device__ bool storeWords(const unsigned int tempOut, int count, int k, unsigned int* C)
{
  while (count > ((1 << 25) - 1))
  {
    count -= (1 << 25) - 1;
    C[k] = tempOut;
    k++;
  }
}

__global__ void generateC(unsigned int m, unsigned int* T1, unsigned int* T2, unsigned int* E, unsigned int* C)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m)
  {
    int count = T1[i];
    int j = count - 1;
    int k = T2[i];
    unsigned int X = E[j];
		
//		printf("tid=%d, j=%d, X=%d, (X & 1)=%d, count_ones=%d, position_of_one(X)=%d\n", i, j, X, (X & 1), count_ones(X, 0, 31), position_of_one(X));
		
    if (i != 0)
      count -= T1[i - 1];

    if ((X & 1) == 1)
    {
      unsigned int tempOut = (33554431 << 7) + (X & 3);
      storeWords(tempOut, count, k, C);
      C[k] = (count << 7) + (X & 3);
    }
    else if (count_ones(X, 0, 31) == 1)
    {
      //      unsigned int p = (32 - log2((float) (X >> 1))) - 1;
      unsigned int p = position_of_one(X) + 1;
      unsigned int tempOut = (33554431 << 7) + 1;
      storeWords(tempOut, count, k, C);
//			printf("tid=%d, count=%d, p=%d\n", i, count, p);
      C[k] = (count << 7) + (p << 2) + 1; ///!!!must use parenthesis, + has high priority than <<
//			printf("tid=%d, count=%d, p=%d, k=%d, C[k]=%d\n", i, count, p, k, C[k]);
    }
    else if (count_ones(X, 0, 31) == 30)
    {
      //      unsigned int p = (32 - log2((float) ((~X) >> 1))) - 1;
      unsigned int p = position_of_one(~X) + 1;
      unsigned int tempOut = (33554431 << 7) + 3;
      storeWords(tempOut, count, k, C);
      C[k] = (count << 7) + (p << 2) + 3;
//			printf("tid=%d, count=%d, p=%d, k=%d, C[k]=%d\n", i, count, p, k, C[k]);
    }
    else
    {
      C[k] = X;
    }

  }
}

__global__ void initS(unsigned int m, unsigned int* C, unsigned int* S)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < m)
  {
		unsigned int c = C[tid];
    if ( (c & 1) == 0)
		{
      S[tid] = 1;
		}
		else
		{
			unsigned int count = c >> 7;
//			if( (c >> 2) & 31 )
//				count++;
			S[tid] = count;
		}
  }
}

__global__ void decomp_initF(unsigned int m, unsigned int* SS, unsigned int* F)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < m && tid > 0)
  {
		F[SS[tid] - 1] = 1;
	}
}

__global__ void generateE(unsigned int n, unsigned int *SF, unsigned int *C, unsigned int *E)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
		unsigned int D = C[SF[i]];
		if( (D&1) == 0)
			E[i] = D;
		else
		{
			if( (D  & (1 << 1)) == 0 )
				E[i] = 0;
			else
				E[i] = ~1;
			
			if( (SF[i] != SF[i+1] || i == n-1) &&  ((D >> 2) & 31) != 0 )
			{
				
				unsigned int p = ((D >> 2) & 31) - 1;
				unsigned int mask = 1 << (31-p);
				unsigned int oldE = E[i];
				unsigned int maskedE = oldE & mask;
				if(maskedE)
				{
					E[i] = oldE & (~mask);
				}
				else
				{
					E[i] = oldE | mask;
				}
			}
		}
	}
}

#endif /* KERNELS_CUH_ */
