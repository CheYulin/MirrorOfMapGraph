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
 * Compressor.cuh
 *
 *  Created on: Apr 29, 2014
 *      Author: zhisong
 */

#ifndef COMPRESSOR_CUH_
#define COMPRESSOR_CUH_

#include <BitmapCompressor/kernels.cuh>
#include <b40c/util/error_utils.cuh>
#include <bitset>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

using namespace std;

class Compressor
{
  int bitmap_size;
  unsigned int *bitmap_extended;
  unsigned int *bitmap_F;
  unsigned int *bitmap_SF;
  unsigned int *T1;
  unsigned int *T2;
  unsigned int *bitmap_C;
  //  unsigned int *bitmap_wah;
public:

  ~Compressor()
  {
  }

  Compressor(int bitmap_size) :
      bitmap_size(bitmap_size)
  {
    unsigned int word_size = (bitmap_size + 31 - bitmap_size % 31) / 31;

    util::B40CPerror(
        cudaMalloc((void**) &bitmap_extended,
            2 * word_size * sizeof(unsigned int)), // double the wordsize for temperary merged array in bitwise operation
        "CsrProblem cudaMalloc bitmap_extended failed",
        __FILE__, __LINE__);

    util::B40CPerror(cudaMemset(bitmap_extended, 0, word_size * sizeof(unsigned int)),
        "Memset bitmap_extended failed", __FILE__, __LINE__);

    util::B40CPerror(
        cudaMalloc((void**) &bitmap_F,
            word_size * sizeof(unsigned int)),
        "CsrProblem cudaMalloc bitmap_F failed",
        __FILE__, __LINE__);

    util::B40CPerror(
        cudaMalloc((void**) &bitmap_SF,
            word_size * sizeof(unsigned int)),
        "CsrProblem cudaMalloc bitmap_SF failed",
        __FILE__, __LINE__);

    util::B40CPerror(
        cudaMalloc((void**) &T1,
            word_size * sizeof(unsigned int)),
        "CsrProblem cudaMalloc T1 failed",
        __FILE__, __LINE__);

    util::B40CPerror(
        cudaMalloc((void**) &T2,
            word_size * sizeof(unsigned int)),
        "CsrProblem cudaMalloc T2 failed",
        __FILE__, __LINE__);

//    util::B40CPerror(
//        cudaMalloc((void**) &bitmap_wah,
//            word_size * sizeof(unsigned int)),
//        "CsrProblem cudaMalloc bitmap_wah failed",
//        __FILE__, __LINE__);
  }

  void compress(unsigned char *bitmap, unsigned int* bitmap_compressed, unsigned int &compressed_size)
  {
    unsigned int threads = 256;
    unsigned int word_size = (bitmap_size + 31 - 1) / 31;
    unsigned int blocks = (word_size + threads - 1) / threads;
    //    unsigned int byte_size = (bitmap_size + 8 - 1) / 8;
    generateExtendedBitmap<< <blocks, threads >> >(bitmap_size, bitmap, bitmap_extended);

    initF<< <blocks, threads >> >(word_size, bitmap_extended, bitmap_F);

    //    unsigned int *out_h = (unsigned int*)malloc(word_size * sizeof (unsigned int));
    //    cudaMemcpy(out_h, bitmap_extended, word_size * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //
    //    printf("bitmap_extended:\n");
    //    for (int i = 0; i < word_size; i++)
    //    {
    //      bitset < 32 > b(out_h[i]);
    //      cout << b << endl;
    //    }

    //    cudaMemcpy(out_h, bitmap_F, word_size * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //
    //    printf("word_size=%d, bitmap_F:\n", word_size);
    //    for (int i = 0; i < word_size; i++)
    //    {
    //      bitset < 32 > b(out_h[i]);
    //      cout << b << endl;
    //    }

    //		thrust::device_ptr<int>  bitmap_F_ptr = thrust::device_pointer_cast(bitmap_F);
    thrust::device_ptr<unsigned int> bitmap_F_ptr(bitmap_F);
    thrust::device_ptr<unsigned int> bitmap_SF_ptr(bitmap_SF);
    //		thrust::device_ptr<unsigned int> bitmap_F_ptr = thrust::device_pointer_cast(bitmap_F);
    thrust::exclusive_scan(bitmap_F_ptr, bitmap_F_ptr + word_size, bitmap_SF_ptr);
    int m;
    cudaMemcpy(&m, bitmap_SF + word_size - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    m += 1;

    //    printf("m=%d, bitmap_SF:\n", m);
    //    cudaMemcpy(out_h, bitmap_SF, word_size * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < word_size; i++)
    //    {
    //      //      bitset < 32 > b(out_h[i]);
    //      cout << out_h[i] << endl;
    //    }

    initT1<< <blocks, threads >> >(word_size, bitmap_F, bitmap_SF, T1);

    //    printf("T1:\n");
    //    cudaMemcpy(out_h, T1, m * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < m; i++)
    //    {
    //      //      bitset < 32 > b(out_h[i]);
    //      cout << out_h[i] << endl;
    //    }

    blocks = (m + threads - 1) / threads;
    initT2<< <blocks, threads >> >(m, T1, T2);

    //    printf("T2:\n");
    //    cudaMemcpy(out_h, T2, m * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < m; i++)
    //    {
    //      cout << out_h[i] << endl;
    //    }

    unsigned int rs;
    cudaMemcpy(&rs, T2 + m - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    thrust::device_ptr<unsigned int> T2_ptr(T2);
    thrust::exclusive_scan(T2_ptr, T2_ptr + m, T2_ptr);
    unsigned int rs2;
    cudaMemcpy(&rs2, T2 + m - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    rs += rs2;

    //    printf("word_size=%d, rs=%d, T2:\n", word_size, rs);
    //    cudaMemcpy(out_h, T2, m * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < m; i++)
    //    {
    //      cout << out_h[i] << endl;
    //    }

    generateC<< <blocks, threads >> >(m, T1, T2, bitmap_extended, bitmap_compressed);

    //    printf("rs=%d, bitmap_compressed:\n", rs);
    //    cudaMemcpy(out_h, bitmap_compressed, rs * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < rs; i++)
    //    {
    //      bitset < 32 > b(out_h[i]);
    //      cout << b << endl;
    //    }

    util::B40CPerror(cudaDeviceSynchronize(),
        "Compression failed", __FILE__, __LINE__);
    compressed_size = rs * sizeof(unsigned int);
  }

  void decompress(int m, unsigned int *bitmap_compressed, unsigned char* bitmap, unsigned int &decompressed_size)
  {

    m /= sizeof(unsigned int);//m is the number of byte in compressed bitmap
    unsigned int* S = bitmap_F;
    unsigned int* SS = bitmap_SF;

    //		unsigned int *out_h = (unsigned int*)malloc(m * sizeof (unsigned int));

    unsigned int threads = 256;
    unsigned int blocks = (m + threads - 1) / threads;
    initS<< <blocks, threads >> >(m, bitmap_compressed, S);

    //		printf("m=%d, S:\n", m);
    //    cudaMemcpy(out_h, S, m * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < m; i++)
    //    {
    //      bitset < 32 > b(out_h[i]);
    //      cout << b << endl;
    //    }

    thrust::device_ptr<unsigned int> S_ptr(S);
    thrust::device_ptr<unsigned int> SS_ptr(SS);
    thrust::exclusive_scan(S_ptr, S_ptr + m, SS_ptr);
    unsigned int S_last, SS_last;
    cudaMemcpy(&S_last, S + m - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&SS_last, SS + m - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int n = S_last + SS_last;
//    decompressed_size = n;

    //		free(out_h);
    unsigned int *out_h = (unsigned int*) malloc(n * sizeof(unsigned int));

    //		printf("n=%d, SS:\n", n);
    //    cudaMemcpy(out_h, SS, m * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < m; i++)
    //    {
    //      bitset < 32 > b(out_h[i]);
    //      cout << b << endl;
    //    }

    thrust::fill(S_ptr, S_ptr + m, 0);
    unsigned int* F = S;
    decomp_initF<< <blocks, threads >> >(m, SS, F);

    //		printf("n=%d, F:\n", n);
    //    cudaMemcpy(out_h, F, n * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < m; i++)
    //    {
    //      bitset < 32 > b(out_h[i]);
    //      cout << b << endl;
    //    }

    unsigned int* SF = SS;
    thrust::device_ptr<unsigned int> F_ptr(F);
    thrust::device_ptr<unsigned int> SF_ptr(SF);
    thrust::exclusive_scan(F_ptr, F_ptr + m, SF_ptr);

    //		printf("n=%d, SF:\n", n);
    //    cudaMemcpy(out_h, SF, n * sizeof (unsigned int), cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < n; i++)
    //    {
    //      bitset < 32 > b(out_h[i]);
    //      cout << b << endl;
    //    }

    blocks = (n + threads - 1) / threads;
    generateE<< <blocks, threads >> >(n, SF, bitmap_compressed, bitmap_extended);

    util::B40CPerror(cudaDeviceSynchronize(),
        "Decompression failed", __FILE__, __LINE__);

//    printf("n=%d, bitmap_extended:\n", n);
//    cudaMemcpy(out_h, bitmap_extended, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < n; i++)
//    {
//      bitset < 32 > b(out_h[i]);
//      cout << b << endl;
//    }

    int b_size = (n * 31 + 7) / 8;
    blocks = (b_size + threads - 1) / threads;
    generateBitmapFromExtended<< <blocks, threads >> >(b_size, bitmap_extended, (unsigned char*)bitmap);
    decompressed_size = b_size;

//    printf("b_size=%d, bitmap:\n", b_size);
//    unsigned char *out_h2 = (unsigned char*) malloc(word_size);
//    cudaMemcpy(out_h2, bitmap, word_size, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < b_size; i++)
//    {
//      bitset < 32 > b(out_h2[i]);
//      cout << b << endl;
//    }

//    bool correct = true;
//    for (int i = 0; i < word_size; i++)
//    {
//      if (bitmap_original[i] != out_h2[i])
//      {
//        printf("byte %d is not the same: bitmap_original=%d, bitmap=%d\n", i, bitmap_original[i], out_h2[i]);
//        correct = false;
//      }
//    }
//
//    if (correct)
//      printf("word_size=%d, b_size=%d, original_size=%d, Compression Correct!!\n", word_size, b_size, original_size);
//    else
//      printf("word_size=%d, b_size=%d, original_size=%d, Compression Wrong!!\n", word_size, b_size, original_size);

//    unsigned int *out_h2 = (unsigned int*) malloc(n * sizeof(unsigned int));
//    cudaMemcpy(out_h2, bitmap_extended, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//
//    bool correct = true;
//    for (int i = 0; i < n; i++)
//    {
//      if ((out_h[i] >> 1) != (out_h2[i] >> 1))
//        correct = false;
//    }

    //    if (correct)
    //      printf("Compression Correct!!\n");
    //    else
    //      printf("Compression Wrong!!\n");

//    free(out_h2);
  }
};

#endif /* COMPRESSOR_CUH_ */

