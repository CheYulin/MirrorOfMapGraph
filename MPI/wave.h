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

#include "mpi.h"
#include "kernel.cuh"
#include <GASengine/statistics.h>
#include <BitmapCompressor/Compressor.cuh>
#include <iostream>
#include <bitset>
#ifndef WAVE_H_
#define WAVE_H_
using namespace std;
using namespace MPI;
using namespace mpikernel;

class wave
//frontier contraction in a 2-d partitioned graph
{
public:
  int pi; //row
  int pj; //column
  int p;
  int n;
  MPI_Group orig_group, new_row_group, new_col_group;
  MPI_Comm new_row_comm, new_col_comm;
  int new_row_rank, new_col_rank;
  double init_time, propagate_time, broadcast_time, compression_time, copy_time, bitunion_time, decompression_time, propagate_wait, broadcast_wait;
  double prop_row, prop_col;

  Statistics* stats;

  unsigned int *bitmap_compressed;
  unsigned char *bitmap_decompressed;
  unsigned char *out_copy, *assigned_temp, *prefix_temp;
  Compressor* comp;
  double compression_ratio_broadcast;
  double compression_ratio;
  unsigned int compressed_size;
  public:

  wave(int l_pi, int l_pj, int l_p, int l_n, Statistics* l_stats) :
      init_time(0.0), compressed_size(0), propagate_time(0.0), propagate_wait(
          0.0), broadcast_time(0.0), broadcast_wait(0.0), compression_time(
          0.0), copy_time(0.0), bitunion_time(0.0), prop_row(0.0), prop_col(0.0)

  //l_pi is the x index
  //l_pj is the y index
  //l_p  is the number of partitions in 1d. usually, sqrt(number of processors)
  //l_n  is the size of the problem, number of vertices
  {

    double starttime, endtime;
    starttime = MPI_Wtime();
    pi = l_pi;
    pj = l_pj;
    p = l_p;
    n = l_n;
    stats = l_stats;

    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    //build original ranks for the processors

    //		int row_indices[p], col_indices[p + 1];
    int *row_indices = new int[p];
    int *col_indices = new int[p + 1];

    for (int i = 0; i < p; i++)
      row_indices[i] = pi * p + i;
    /*		for(int i=0;i<=pi-1;i++)
     row_indices[i+p] = i*p+pi;
     for(int i=pi+1;i<p;i++)
     row_indices[i+p-1] = i*p+pi;
     */for (int i = 0; i < p; i++)
      col_indices[i] = i * p + pj;
    /*              for(int i=0;i<=pj-1;i++)
     col_indices[i] = i*p+pj;
     for(int i=pj+1;i<p;i++)
     col_indices[i-1] = i*p+pj;
     col_indices[p-1] = pj*p+p-1;
     */
    MPI_Group_incl(orig_group, p, row_indices, &new_row_group);
    MPI_Group_incl(orig_group, p, col_indices, &new_col_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_row_group, &new_row_comm);
    MPI_Comm_create(MPI_COMM_WORLD, new_col_group, &new_col_comm);
    MPI_Group_rank(new_row_group, &new_row_rank);
    MPI_Group_rank(new_col_group, &new_col_rank);
    endtime = MPI_Wtime();
    //MPI_Barrier(new_row_comm);
    //MPI_Barrier(new_col_comm);
    init_time = endtime - starttime;
    propagate_time = 0;
    propagate_wait = 0;
    broadcast_time = 0;
    broadcast_wait = 0;
    copy_time = 0;
    bitunion_time = 0;

    util::B40CPerror(
        cudaMalloc((void**) &out_copy,
            ceil(n / 8.0) * sizeof(unsigned char)));

    util::B40CPerror(
        cudaMalloc((void**) &assigned_temp,
            ceil(n / 8.0) * sizeof(unsigned char)));

    util::B40CPerror(
        cudaMalloc((void**) &prefix_temp,
            ceil(n / 8.0) * sizeof(unsigned char)));

    util::B40CPerror(
        cudaMalloc((void**) &bitmap_compressed,
            (n + 31 - 1) / 31 * sizeof(unsigned int)),
        "CsrProblem cudaMalloc bitmap_compressed failed", __FILE__,
        __LINE__);

    util::B40CPerror(
        cudaMemset(bitmap_compressed, 0,
            (n + 31 - 1) / 31 * sizeof(unsigned int)),
        "Memset bitmap_compressed failed", __FILE__, __LINE__);

    util::B40CPerror(
        cudaMalloc((void**) &bitmap_decompressed,
            (n + 31 - 1) / 31 * sizeof(unsigned int)),
        "CsrProblem cudaMalloc bitmap_decompressed failed", __FILE__,
        __LINE__);

    util::B40CPerror(
        cudaMemset(bitmap_decompressed, 0,
            (n + 31 - 1) / 31 * sizeof(unsigned int)),
        "Memset bitmap_decompressed failed", __FILE__, __LINE__);

    comp = new Compressor(n);
    //    comp = new Compressor(186);

  }

  /*
   void propogate(unsigned char* out_d, unsigned char* assigned_d,
   unsigned char* prefix_d)
   //wave propogation, in sequential from top to bottom of the column
   {
   double starttime, endtime;

   unsigned int mesg_size = ceil(n / 8.0);
   int myid = pi * p + pj;

   int numthreads = 512;
   int byte_size = (n + 8 - 1) / 8;
   int numblocks = min(512, (byte_size + numthreads - 1) / numthreads);


   MPI_Request request[2];
   MPI_Status status[2];
   if (p > 1)
   {

   if (pj == 0)
   {
   starttime = MPI_Wtime();
   MPI_Isend(out_d, mesg_size, MPI_CHAR, myid + 1, pi,
   MPI_COMM_WORLD, &request[1]);
   MPI_Wait(&request[1], &status[1]);


   endtime = MPI_Wtime();
   propagate_time = endtime - starttime;
   }

   else if (pj != p - 1)
   {
   starttime = MPI_Wtime();

   MPI_Irecv(prefix_d, mesg_size, MPI_CHAR, myid - 1, pi, MPI_COMM_WORLD, &request[0]);
   MPI_Wait(&request[0], &status[0]);
   endtime = MPI_Wtime();
   propagate_time = endtime - starttime;

   starttime = MPI_Wtime();
   mpikernel::bitunion << <numblocks, numthreads >> >(mesg_size, out_d, prefix_d, out_d);
   cudaDeviceSynchronize();
   endtime = MPI_Wtime();
   bitunion_time = endtime - starttime;

   starttime = MPI_Wtime();
   MPI_Isend(out_d, mesg_size, MPI_CHAR, myid + 1, pi, MPI_COMM_WORLD, &request[1]);


   MPI_Wait(&request[1], &status[1]);

   endtime = MPI_Wtime();
   propagate_time += endtime - starttime;
   }

   else
   {
   starttime = MPI_Wtime();

   MPI_Irecv(prefix_d, mesg_size, MPI_CHAR, myid - 1, pi,
   MPI_COMM_WORLD, &request[0]);
   MPI_Wait(&request[0], &status[0]);

   endtime = MPI_Wtime();
   propagate_time = endtime - starttime;
   mpikernel::bitunion << <numblocks, numthreads >> >(mesg_size, out_d, prefix_d, out_d);
   cudaDeviceSynchronize();
   }
   }

   } */

  void propogate_tree(unsigned char* out_d, unsigned char* assigned_d,
      unsigned char* prefix_d)
  {
    //printf("tree");

    int distance = 1;
    bitunion_time = 0.0;
    unsigned int mesg_size = ceil(n / 8.0);
    int rank_id = pi * p + pj;
    cudaMemcpy(out_copy, out_d, mesg_size, cudaMemcpyDeviceToDevice);
    double starttime, endtime, startbitunion, endbitunion;
    double waitstart, waitend;
    int numthreads = 256;
    int numblocks = (mesg_size + numthreads - 1) / numthreads;
    MPI_Request request;
    MPI_Status status;
    starttime = MPI_Wtime();
    //    if (rank_id == 0 || rank_id == 1 || rank_id == 2 || rank_id == 3)
    //    {
    //      unsigned int *out_h = (unsigned int*)malloc(mesg_size);
    //      printf("myid=%d, m=%d, initial out_d:\n", rank_id, mesg_size);
    //      cudaMemcpy(out_h, out_d, mesg_size, cudaMemcpyDeviceToHost);
    //      for (int i = 0; i < mesg_size; i++)
    //      {
    //        bitset < 8 > b(out_h[i]);
    //        cout << b << endl;
    //      }
    //      free(out_h);
    //    }

    //    cudaMemcpy(assigned_d, out_d, mesg_size, cudaMemcpyDeviceToDevice);
    propagate_wait = 0.0;
    while (distance < p)
    {
      if ((pj + distance) < p)
      {
        MPI_Isend(out_d, mesg_size, MPI_CHAR, rank_id + distance, pi,
            MPI_COMM_WORLD, &request);
        //        waitstart = MPI_Wtime();
        //        MPI_Wait(&request, &status);
        //        waitend = MPI_Wtime();
        //        propagate_wait += waitend - waitstart;
      }
      if ((pj - distance) >= 0)
      {
        MPI_Irecv(prefix_temp, mesg_size, MPI_CHAR, rank_id - distance,
            pi, MPI_COMM_WORLD, &request);
        waitstart = MPI_Wtime();
        MPI_Wait(&request, &status);
        waitend = MPI_Wtime();
        propagate_wait += waitend - waitstart;
        //        value += x;
        startbitunion = MPI_Wtime();
        mpikernel::bitunion<< <numblocks, numthreads >> >(mesg_size, out_d,
            prefix_temp, out_d);
        mpikernel::bitunion<< <numblocks, numthreads >> >(mesg_size,
            prefix_temp, prefix_d, prefix_d);
        cudaDeviceSynchronize();
        endbitunion = MPI_Wtime();
        bitunion_time += endbitunion - startbitunion;
      }
      distance *= 2;
      //      MPI_Barrier(MPI_COMM_WORLD);
      //      if (rank_id == 3)
      //      {
      //        unsigned int *out_h = (unsigned int*)malloc(mesg_size);
      //        printf("myid=%d, m=%d, distance=%d, out_d:\n", rank_id, mesg_size, distance);
      //        cudaMemcpy(out_h, out_d, mesg_size, cudaMemcpyDeviceToHost);
      //        for (int i = 0; i < mesg_size; i++)
      //        {
      //          bitset < 8 > b(out_h[i]);
      //          cout << b << endl;
      //        }
      //        free(out_h);
      //      }
    }

    //    mpikernel::bitsubstract<<<numblocks, numthreads>>>(mesg_size, out_d,
    //        out_copy, prefix_d);
    startbitunion = MPI_Wtime();
    mpikernel::bitsubstract<< <numblocks, numthreads >> >(mesg_size, out_copy,
        prefix_d, assigned_temp);
    mpikernel::bitunion<< <numblocks, numthreads >> >(mesg_size, assigned_d,
        assigned_temp, assigned_d);
    endbitunion = MPI_Wtime();
    endtime = MPI_Wtime();
    propagate_time = endtime - starttime;
    bitunion_time += endbitunion - startbitunion;
  }

  void propogate(unsigned char* out_d, unsigned char* assigned_d,
      unsigned char* prefix_d)
  //wave propogation, in sequential from top to bottom of the column
  {
    double starttime, endtime;
    unsigned int mesg_size = ceil(n / 8.0);
    int myid = pi * p + pj;

    int numthreads = 512;
    int byte_size = (n + 8 - 1) / 8;
    int numblocks = (byte_size + numthreads - 1) / numthreads;

    MPI_Request request[2];
    MPI_Status status[2];
    //    MPI_Barrier(MPI_COMM_WORLD);
    if (p > 1)
    {

      if (pj == 0)
      {
        starttime = MPI_Wtime();
        MPI_Send(out_d, mesg_size, MPI_CHAR, myid + 1, pi,
            MPI_COMM_WORLD);
        endtime = MPI_Wtime();
        propagate_time = endtime - starttime;
      }

      else if (pj != p - 1)
      {
        starttime = MPI_Wtime();

        MPI_Recv(prefix_d, mesg_size, MPI_CHAR, myid - 1, pi,
            MPI_COMM_WORLD, &status[1]);
        //MPI_Wait(&request[0], &status[0]);
        endtime = MPI_Wtime();
        propagate_time = endtime - starttime;

        starttime = MPI_Wtime();
        mpikernel::bitunion<< <numblocks, numthreads >> >(mesg_size, out_d,
            prefix_d, out_d);
        cudaDeviceSynchronize();
        endtime = MPI_Wtime();
        bitunion_time = endtime - starttime;

        starttime = MPI_Wtime();
        MPI_Send(out_d, mesg_size, MPI_CHAR, myid + 1, pi,
            MPI_COMM_WORLD);

        //MPI_Wait(&request[1], &status[1]);

        endtime = MPI_Wtime();
        propagate_time += endtime - starttime;
      }

      else
      {
        starttime = MPI_Wtime();

        MPI_Recv(prefix_d, mesg_size, MPI_CHAR, myid - 1, pi,
            MPI_COMM_WORLD, &status[1]);
        //MPI_Wait(&request[0], &status[0]);

        endtime = MPI_Wtime();
        propagate_time = endtime - starttime;
        mpikernel::bitunion<< <numblocks, numthreads >> >(mesg_size, out_d,
            prefix_d, out_d);
        cudaDeviceSynchronize();
      }
    }

  }

  void correct_test(unsigned char* tmp1_h, unsigned char* tmp2_h,
      int mesg_size)
  {
    int myid = pi * p + pj;
    bool correct = true;
    for (int i = 0; i < mesg_size; i++)
    {
      //      bitset < 8 > b1(tmp1_h[i]);
      //      if (myid == 0)
      //        cout << b1 << endl;
      //      bitset < 8 > b2(tmp2_h[i]);
      if (tmp1_h[i] != tmp2_h[i])
        correct = false;
    }

    if (correct == false)
      printf("myid: %d, Decompression error!!\n", myid);
    else
      printf("myid: %d, Decompression correct!!\n", myid);

  }

  void propogate_compressed(unsigned char* out_d, unsigned char* assigned_d,
      unsigned char* prefix_d)
  {
    double starttime, endtime;

    //byte number, NOT int number
    unsigned int decompressed_size;

    //    MPI_Barrier(MPI_COMM_WORLD);
    //    starttime = MPI_Wtime();
    unsigned int mesg_size = ceil(n / 8.0);
    int myid = pi * p + pj;
    //int lastid = pi*p+p-1;
    int numthreads = 512;
    int byte_size = (n + 8 - 1) / 8;
    int numblocks = min(512, (byte_size + numthreads - 1) / numthreads);

    //    unsigned char *tmp1_h = (unsigned char*)malloc(mesg_size);
    //    unsigned char *tmp2_h = (unsigned char*)malloc(mesg_size);
    double compress_start;
    double compress_end;
    compression_time = 0.0;

    MPI_Request request[2];
    MPI_Status status[2];
    int tag = 0;
    if (p > 1)
    {
      //if first one in the column, initiate the wave propogation
      if (pj == 0)
      {
        //        cudaMemcpy(tmp1_h, out_d, mesg_size, cudaMemcpyDeviceToHost);

        compress_start = MPI_Wtime();
        comp->compress(out_d, bitmap_compressed, compressed_size);
        compress_end = MPI_Wtime();
        compression_time = compress_end - compress_start;

        //        if (myid == 0)
        //        {
        //          unsigned int *out_h = (unsigned int*)malloc(compressed_size / 4 * sizeof (unsigned int));
        //          printf("myid=%d, m=%d, bitmap_compressed0:\n", myid, compressed_size / 4);
        //          cudaMemcpy(out_h, bitmap_compressed, compressed_size / 4 * sizeof (unsigned int), cudaMemcpyDeviceToHost);
        //          for (int i = 0; i < compressed_size / 4; i++)
        //          {
        //            bitset < 32 > b(out_h[i]);
        //            cout << b << endl;
        //          }
        //        }
        //
        //        printf("myid=%d, compressed_size=%d, n=%d\n", myid, compressed_size, n);

        starttime = MPI_Wtime();
        MPI_Send(bitmap_compressed, compressed_size, MPI_BYTE, myid + 1,
            tag, MPI_COMM_WORLD);
        endtime = MPI_Wtime();
        propagate_time = endtime - starttime;

        //        comp->decompress(compressed_size, bitmap_compressed,
        //                         out_d, decompressed_size);
        //
        //        cudaMemcpy(tmp2_h, out_d, mesg_size, cudaMemcpyDeviceToHost);
        //
        //        correct_test(tmp1_h, tmp2_h, mesg_size);

        //        MPI_Wait(&request[1], &status[1]);
        //free(out_h);
      }
      //else if not the last one, receive bitmap from top, process and send to next one
      else if (pj != p - 1)
      {
        // char *prefix_h = (char*)malloc(mesg_size);

        //        util::B40CPerror(
        //                         cudaMemset(bitmap_compressed, 0,
        //                                    (n + 31 - 1) / 31 * sizeof (unsigned int)),
        //                         "Memset bitmap_compressed failed", __FILE__, __LINE__);

        starttime = MPI_Wtime();
        int word_size = (n + 30) / 31;
        MPI_Probe(myid - 1, tag, MPI_COMM_WORLD, &status[0]);
        MPI_Get_count(&status[0], MPI_BYTE, (int*) &compressed_size);
        MPI_Recv(bitmap_compressed, compressed_size, MPI_BYTE, myid - 1,
            tag, MPI_COMM_WORLD, &status[0]);
        //        MPI_Wait(&request[0], &status[0]);
        //MPI_Get_count(&status[0], MPI_BYTE, (int*)&compressed_size);
        endtime = MPI_Wtime();
        propagate_time = endtime - starttime;

        //        printf("myid=%d, compressed_size=%d, n=%d\n", myid, compressed_size, n);

        compress_start = MPI_Wtime();
        comp->decompress(compressed_size, bitmap_compressed,
            bitmap_decompressed, decompressed_size);
        cudaDeviceSynchronize();
        compress_end = MPI_Wtime();
        compression_time = compress_end - compress_start;

        //        cudaMemcpy(prefix_d, prefix_h, mesg_size, cudaMemcpyHostToDevice);
        //        mpikernel::bitsubstract << <numblocks, numthreads >> >(mesg_size, out_d, prefix_d, assigned_d);
        //        cudaDeviceSynchronize();
        starttime = MPI_Wtime();
        mpikernel::bitunion<< <numblocks, numthreads >> >(byte_size, out_d,
            bitmap_decompressed, out_d);
        cudaDeviceSynchronize();
        endtime = MPI_Wtime();
        bitunion_time = endtime - starttime;
        //cudaMemcpy(out_h, out_d, mesg_size, cudaMemcpyDeviceToHost);

        compress_start = MPI_Wtime();
        comp->compress(out_d, bitmap_compressed, compressed_size);
        cudaDeviceSynchronize();
        compress_end = MPI_Wtime();
        compression_time += compress_end - compress_start;

        //        cudaMemcpy(tmp1_h, out_d, mesg_size, cudaMemcpyDeviceToHost);
        //
        //        comp->decompress(compressed_size, bitmap_compressed,
        //                         out_d, decompressed_size);
        //
        //        cudaMemcpy(tmp2_h, out_d, mesg_size, cudaMemcpyDeviceToHost);
        //
        //        correct_test(tmp1_h, tmp2_h, mesg_size);

        starttime = MPI_Wtime();
        MPI_Send(bitmap_compressed, compressed_size, MPI_BYTE, myid + 1,
            tag, MPI_COMM_WORLD);
        endtime = MPI_Wtime();
        propagate_time += endtime - starttime;
        //f//ree(prefix_h);

        //        MPI_Wait(&request[1], &status[1]);
        //free(out_h);
      }
      //else receive from the previous and then broadcast to the broadcast group
      else
      {
        //        char *prefix_h = (char*)malloc(mesg_size);
        starttime = MPI_Wtime();

        MPI_Probe(myid - 1, tag, MPI_COMM_WORLD, &status[0]);
        MPI_Get_count(&status[0], MPI_BYTE, (int*) &compressed_size);
        MPI_Recv(bitmap_compressed, compressed_size, MPI_BYTE, myid - 1,
            tag, MPI_COMM_WORLD, &status[0]);

        //int word_size = (n + 30) / 31;
        //MPI_Recv(bitmap_compressed, word_size * sizeof (unsigned int),
        //         MPI_BYTE, myid - 1, tag, MPI_COMM_WORLD, &status[0]);

        //MPI_Get_count(&status[0], MPI_BYTE, (int*)&compressed_size);
        endtime = MPI_Wtime();
        propagate_time = endtime - starttime;

        compress_start = MPI_Wtime();
        comp->decompress(compressed_size, bitmap_compressed,
            bitmap_decompressed, decompressed_size);
        cudaDeviceSynchronize();
        compress_end = MPI_Wtime();
        compression_time = compress_end - compress_start;
        //        MPI_Wait(&request[0], &status[0]);
        //cudaMemcpy(prefix_d, prefix_h, mesg_size, cudaMemcpyHostToDevice);
        //mpikernel::bitsubstract << <numblocks, numthreads >> >(mesg_size, out_d, prefix_d, assigned_d);
        //cudaDeviceSynchronize();
        starttime = MPI_Wtime();
        mpikernel::bitunion<< <numblocks, numthreads >> >(mesg_size, out_d,
            (unsigned char*)bitmap_decompressed, out_d);
        cudaDeviceSynchronize();
        endtime = MPI_Wtime();
        bitunion_time = endtime - starttime;
        compressed_size = 0;
      }
    }

    //    endtime = MPI_Wtime();
    //    propagate_time = endtime - starttime - compression_time;
  }

  void broadcast_new_frontier_compressed(unsigned char* out_d,
      unsigned char* in_d)
  {
    double starttime, endtime;

    unsigned int mesg_size = ceil(n / (8.0));
    int myid = pi * p + pj;

    //    unsigned char *out_h = (unsigned char*)malloc(mesg_size);
    //    unsigned char *in_h = (unsigned char*)malloc(mesg_size);
    unsigned int compressed_size;
    unsigned int decompressed_size;

    MPI_Barrier (MPI_COMM_WORLD);
    starttime = MPI_Wtime();
    if (pj == p - 1)
    {
      //      starttime = MPI_Wtime();
      comp->compress(out_d, bitmap_compressed, compressed_size);
      //      endtime = MPI_Wtime();
      //      compression_time = endtime - starttime;
      //      comp->decompress(compressed_size, bitmap_compressed, bitmap_decompressed, decompressed_size);
      //      cudaMemcpy(out_h, out_d, mesg_size, cudaMemcpyDeviceToHost);
      //      cudaMemcpy(in_h, bitmap_decompressed, mesg_size, cudaMemcpyDeviceToHost);
      //      if (pi == 0 && pj == p-1)
      //      {
      //        bool correct = true;
      //        for (int i = 0; i < mesg_size; i++)
      //        {
      //					bitset < 8 > b1(out_h[i]);
      //					bitset < 8 > b2(in_h[i]);
      ////          cout << b1 << "   " << b2 << endl;
      //          if (out_h[i] != in_h[i])
      //            correct = false;
      //        }
      //        if(correct == false) printf("myid: %d, Decompression error!!\n", myid);
      //				else printf("myid: %d, Decompression correct!!\n", myid);
      //      }
      //      cudaMemcpy(in_h, bitmap_decompressed, mesg_size, cudaMemcpyDeviceToHost);
      //      if (pi == 0 && pj == p-1)
      //      {
      //        bool correct = true;
      //        for (int i = 0; i < mesg_size; i++)
      //        {
      //					bitset < 8 > b1(out_h[i]);
      //					bitset < 8 > b2(in_h[i]);
      ////          cout << b1 << "   " << b2 << endl;
      //          if (out_h[i] != in_h[i])
      //            correct = false;
      //        }
      //        if(correct == false) printf("myid: %d, Decompression error!!\n", myid);
      //				else printf("myid: %d, Decompression correct!!\n", myid);
      //      }
    }

    MPI_Bcast(&compressed_size, 1, MPI_UNSIGNED, p - 1, new_row_comm);
    MPI_Bcast(bitmap_compressed, compressed_size, MPI_BYTE, p - 1,
        new_row_comm);

    comp->decompress(compressed_size, bitmap_compressed, out_d,
        decompressed_size);
    //		MPI_Bcast(bitmap_compressed, 8, MPI_BYTE, p - 1, new_row_comm);
    //    cudaMemcpy(out_d, out_h, mesg_size, cudaMemcpyHostToDevice);

    //using bitmap_decompressed as temp buffer
    //    if (pi == pj)
    //      cudaMemcpy(bitmap_decompressed, bitmap_compressed, compressed_size, cudaMemcpyDeviceToDevice);

    //    unsigned int compressed_size2 = compressed_size;
    MPI_Bcast(&compressed_size, 1, MPI_UNSIGNED, pj, new_col_comm);
    MPI_Bcast(bitmap_compressed, compressed_size, MPI_BYTE, pj,
        new_col_comm);
    //		MPI_Bcast(bitmap_compressed, 8, MPI_BYTE, pj, new_col_comm);

    comp->decompress(compressed_size, bitmap_compressed, in_d,
        decompressed_size);

    //    cudaMemcpy(out_d, out_h, mesg_size, cudaMemcpyHostToDevice);
    //    cudaMemcpy(in_d, in_h, mesg_size, cudaMemcpyHostToDevice);

    //    cudaDeviceSynchronize();
    //    free(in_h);
    //    free(out_h);
    endtime = MPI_Wtime();
    broadcast_time = endtime - starttime;

    compression_ratio_broadcast = (double) compressed_size / mesg_size;
    //    if (pj==p-1)
    //			printf("myid: %d compressed_size: %d original_size: %d compression_ratio_broadcast: %lf\n",
    //												myid, compressed_size, mesg_size, compression_ratio_broadcast);
    //		printf("myid: %d broadcast_time: %lf\n", myid, broadcast_time);
  }

  //Version that does not support GPUDirect

  void reduce_frontier_CPU(unsigned char* out_d, unsigned char* in_d)
  {
    double starttime, endtime;

    unsigned int mesg_size = ceil(n / (8.0));
    unsigned int word_size = (n + 30) / 31;
    unsigned char *out_h = (unsigned char*) malloc(
        word_size * sizeof(unsigned int));
    unsigned char *out_h2 = (unsigned char*) malloc(mesg_size);
    unsigned char *in_h = (unsigned char*) malloc(mesg_size);
    cudaMemcpy(out_h, out_d, word_size * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);

    //		for(int i=0; i<mesg_size; i++)
    //		{
    //			out_h[i] = 0;
    //		}
    //		out_h[6] = 1 << 5;
    //		out_h[16] = 1 << 4;
    //		out_h[21] = 1 << 3;
    //
    //		cudaMemcpy(out_d, out_h, 24, cudaMemcpyHostToDevice);

    //		printf("bitmap_out_d:\n");
    //    for (int i = 0; i < 24; i++)
    //    {
    //      bitset < 8 > b(out_h[i]);
    //      cout << b << endl;
    //    }
    unsigned int compressed_size; //number of bytes
    unsigned int decompressed_size; //number of bytes

    starttime = MPI_Wtime();
    comp->compress(out_d, bitmap_compressed, compressed_size);
    endtime = MPI_Wtime();
    compression_time = endtime - starttime;

    starttime = MPI_Wtime();
    comp->decompress(compressed_size, bitmap_compressed,
        bitmap_decompressed, decompressed_size);
    endtime = MPI_Wtime();
    decompression_time = endtime - starttime;

    starttime = MPI_Wtime();
    MPI_Allreduce(out_h, out_h2, mesg_size, MPI_BYTE, MPI_BOR,
        new_row_comm);

    cudaMemcpy(out_d, out_h2, mesg_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    endtime = MPI_Wtime();
    propagate_time = endtime - starttime;

    compression_ratio = (double) compressed_size / decompressed_size;

    starttime = MPI_Wtime();
    if (pi == pj)
      memcpy(in_h, out_h2, mesg_size);

    MPI_Bcast(in_h, mesg_size, MPI_CHAR, pj, new_col_comm);
    endtime = MPI_Wtime();
    broadcast_time = endtime - starttime;

    cudaMemcpy(in_d, in_h, mesg_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(in_h);
    free(out_h);

  }

  //version that supports GPUDirect

  void reduce_frontier_GDR(unsigned char* out_d, unsigned char* in_d)
  {

    unsigned int mesg_size = ceil(n / (8.0));
    //		unsigned char *out_h = (unsigned char*) malloc(mesg_size);
    //    cudaMemcpy(out_h, out_d, mesg_size, cudaMemcpyDeviceToHost);
    //    for (int i = 0; i < mesg_size; i++)
    //    {
    //      bitset < 8 > b(out_h[i]);
    //      cout << b << endl;
    //    }
    //    comp->compress(out_d, bitmap_compressed);

    double starttime, endtime;
    //    MPI_Barrier(MPI_COMM_WORLD);
    starttime = MPI_Wtime();
    MPI_Allreduce(out_d, out_d, mesg_size, MPI_BYTE, MPI_BOR, new_row_comm);
    //		MPI_Allreduce(out_d, out_d, mesg_size, MPI_BYTE, MPI_BOR, MPI_COMM_WORLD);

    //		MPI_Barrier(MPI_COMM_WORLD);
    endtime = MPI_Wtime();

    propagate_time = endtime - starttime;

    if (pi == pj)
      cudaMemcpy(in_d, out_d, mesg_size, cudaMemcpyDeviceToDevice);

    //    MPI_Barrier(MPI_COMM_WORLD);
    starttime = MPI_Wtime();
    MPI_Bcast(in_d, mesg_size, MPI_CHAR, pj, new_col_comm);
    //		MPI_Bcast(in_d, mesg_size, MPI_CHAR, pj, MPI_COMM_WORLD);
    //		MPI_Barrier(MPI_COMM_WORLD);
    endtime = MPI_Wtime();
    broadcast_time = endtime - starttime;

    //    if (pi == p - 1)
    //    {
    //      int count = 0;
    //      char *in_h = (char*)malloc(mesg_size);
    //      cudaMemcpy(in_h, in_d, mesg_size, cudaMemcpyDeviceToHost);
    //      //#pragma omp parallel for reduction(+:count)
    //      for (int i = 0; i < mesg_size; i++)
    //      {
    //        count += (int)(in_h[i] >> 0 && 1);
    //        count += (int)(in_h[i] >> 1 && 1);
    //        count += (int)(in_h[i] >> 2 && 1);
    //        count += (int)(in_h[i] >> 3 && 1);
    //        count += (int)(in_h[i] >> 4 && 1);
    //        count += (int)(in_h[i] >> 5 && 1);
    //        count += (int)(in_h[i] >> 6 && 1);
    //        count += (int)(in_h[i] >> 7 && 1);
    //      }
    //
    //      printf(" %d", count);
    //    }

  }

  void broadcast_new_frontier_nonblocking(unsigned char* out_d,
      unsigned char* in_d)
  {
    //printf("In Broadcast");
    double starttime, endtime, waitstart, waitend;
    MPI_Request request[2];
    MPI_Status status[2];
    broadcast_time = 0.0;
    broadcast_wait = 0.0;
    //    MPI_Barrier(MPI_COMM_WORLD);
    unsigned int mesg_size = ceil(n / (8.0));
    starttime = MPI_Wtime();
    MPI_Ibcast(out_d, mesg_size, MPI_CHAR, p - 1, new_row_comm,
        &request[0]);

    if (pi == pj)
    {
      waitstart = MPI_Wtime();
      MPI_Wait(&request[0], &status[0]);
      waitend = MPI_Wtime();
      broadcast_wait += waitend - waitstart;

      starttime = MPI_Wtime();
      cudaMemcpy(in_d, out_d, mesg_size, cudaMemcpyDeviceToDevice);
      endtime = MPI_Wtime();
      copy_time = endtime - starttime;
      MPI_Ibcast(in_d, mesg_size, MPI_CHAR, pj, new_col_comm,
          &request[1]);
      waitstart = MPI_Wtime();
      MPI_Wait(&request[1], &status[1]);
      waitend = MPI_Wtime();
      broadcast_wait += waitend - waitstart;
    }
    else
    {
      MPI_Ibcast(in_d, mesg_size, MPI_CHAR, pj, new_col_comm,
          &request[1]);
      waitstart = MPI_Wtime();
      MPI_Wait(&request[0], &status[0]);
      MPI_Wait(&request[1], &status[1]);
      waitend = MPI_Wtime();
      broadcast_wait += waitend - waitstart;
    }

    endtime = MPI_Wtime();
    broadcast_time += endtime - starttime;

  }

  void broadcast_new_frontier(unsigned char* out_d, unsigned char* in_d)
  {
    double starttime, endtime;
    //    MPI_Barrier(MPI_COMM_WORLD);
    unsigned int mesg_size = ceil(n / (8.0));
    starttime = MPI_Wtime();
    MPI_Bcast(out_d, mesg_size, MPI_CHAR, p - 1, new_row_comm);
    endtime = MPI_Wtime();
    broadcast_time = endtime - starttime;

    starttime = MPI_Wtime();
    if (pi == pj)
      cudaMemcpy(in_d, out_d, mesg_size, cudaMemcpyDeviceToDevice);
    endtime = MPI_Wtime();
    copy_time = endtime - starttime;

    starttime = MPI_Wtime();
    MPI_Bcast(in_d, mesg_size, MPI_CHAR, pj, new_col_comm);
    endtime = MPI_Wtime();
    broadcast_time += endtime - starttime;

  }

  void broadcast_tree(unsigned char* out_d, unsigned char* in_d, int iter)
  {
    double starttime, endtime;
    double prop_start, prop_end;
    int rank_id = pi * p + pj;
    MPI_Request request[7];
    MPI_Status status[3];
    broadcast_wait = 0.0;
    broadcast_time = 0.0;
    //        MPI_Barrier(MPI_COMM_WORLD);
    unsigned int mesg_size = ceil(n / (8.0));
    //    int in[2] = { 0, 1 };
    //    int out[2] = { 0, 1 };

    starttime = MPI_Wtime();
    prop_start = MPI_Wtime();
    int seg_size = p;
    while ((seg_size >> 1) > 0)
    {
      if ((pj + 1) % seg_size == 0)
      {
        int recv_rank = p * pi + pj - (seg_size >> 1);
        MPI_Isend(out_d, mesg_size, MPI_CHAR, recv_rank, 0,
            MPI_COMM_WORLD, &request[0]);
//        printf("pi=%d, pj=%d, to=%d\n", pi, pj, recv_rank);
      }

      if ((pj + 1 + (seg_size >> 1)) % seg_size == 0)
      {
        int send_rank = p * pi + pj + (seg_size >> 1);
        MPI_Irecv(out_d, mesg_size, MPI_CHAR, send_rank, 0, MPI_COMM_WORLD, &request[1]);

        MPI_Wait(&request[1], &status[0]);
//        printf("iter=%d, pi=%d, pj=%d, from=%d\n", iter, pi, pj, send_rank);
      }
      seg_size >>= 1;

      //      fflush(stdout);
      //      usleep(1000);
      //      MPI_Barrier(MPI_COMM_WORLD);

    }

    prop_end = MPI_Wtime();
    prop_row = prop_end - prop_start;

    //    cudaDeviceSynchronize();

    //        MPI_Barrier(MPI_COMM_WORLD);

    prop_start = MPI_Wtime();
    if (pj == p - 1)
    {
      int recv_rank = pi;
      MPI_Isend(out_d, mesg_size, MPI_CHAR, recv_rank, 1,
          MPI_COMM_WORLD, &request[2]);

      //      printf("pi=%d, pj=%d, to=%d\n", pi, pj, recv_rank);
      recv_rank = (p >> 1) * p + pi;
      MPI_Isend(out_d, mesg_size, MPI_CHAR, recv_rank, 1,
          MPI_COMM_WORLD, &request[3]);
      //      printf("pi=%d, pj=%d, to=%d\n", pi, pj, recv_rank);
    }

    if (pi == 0 || pi == (p >> 1))
    {
      int send_rank = pj * p + p - 1;
      MPI_Irecv(in_d, mesg_size, MPI_CHAR, send_rank,
          1, MPI_COMM_WORLD, &request[4]);

      //      printf("pi=%d, pj=%d, from=%d\n", pi, pj, send_rank);
      MPI_Wait(&request[4], &status[1]);
    }

    //    MPI_Barrier(MPI_COMM_WORLD);

    seg_size = p >> 1;
    while ((seg_size >> 1) > 0)
    {
      if (pi % seg_size == 0)
      {
        int recv_rank = p * (pi + (seg_size >> 1)) + pj;
        MPI_Isend(in_d, mesg_size, MPI_CHAR, recv_rank, 1,
            MPI_COMM_WORLD, &request[5]);
        //        printf("pi=%d, pj=%d, to=%d\n", pi, pj, recv_rank);
      }

      if ((pi - (seg_size >> 1)) % seg_size == 0)
      {
        int send_rank = (pi - (seg_size >> 1)) * p + pj;
        MPI_Irecv(in_d, mesg_size, MPI_CHAR, send_rank,
            1, MPI_COMM_WORLD, &request[6]);

        //        printf("pi=%d, pj=%d, from=%d\n", pi, pj, send_rank);
        MPI_Wait(&request[6], &status[2]);
      }
      seg_size >>= 1;

      //      fflush(stdout);
      //      usleep(1000);
      //      MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier (MPI_COMM_WORLD);
    prop_end = MPI_Wtime();
    endtime = MPI_Wtime();
    broadcast_time = endtime - starttime;
    prop_col = prop_end - prop_start;

    //    if (pj == p - 1)
    //    {
    //      for (int i = 0; i < p - 1; i++)
    //      {
    //        int recv_rank = pi * p + i;
    //        MPI_Isend(out_d, mesg_size, MPI_CHAR, recv_rank, 0,
    //            MPI_COMM_WORLD, &request[0]);
    //      }
    //
    ////      if (pi != p - 1)
    ////      {
    //      for (int i = 0; i < p; i++)
    //      {
    //        if (i != pi)
    //        {
    //          int recv_rank = i * p + pi;
    //          MPI_Isend(out_d, mesg_size, MPI_CHAR, recv_rank, 1,
    //              MPI_COMM_WORLD, &request[1]);
    ////            MPI_Isend(out, 2, MPI_INT, recv_rank, 1,
    ////                MPI_COMM_WORLD, &request[1]);
    ////            printf("pi=%d, pj=%d, to=%d\n", pi, pj, recv_rank);
    //        }
    //      }
    ////      }
    //
    ////      if (pi == p - 1)
    ////      {
    ////        for (int i = 0; i < p - 1; i++)
    ////        {
    ////          int recv_rank = i * p + pi;
    ////          MPI_Isend(out_d, mesg_size, MPI_CHAR, recv_rank, 1,
    ////              MPI_COMM_WORLD, &request[2]);
    //////          MPI_Isend(out, 2, MPI_INT, recv_rank, 1,
    //////              MPI_COMM_WORLD, &request[2]);
    //////          printf("pi=%d, pj=%d, to=%d\n", pi, pj, recv_rank);
    ////        }
    ////      }
    //
    //      if (pi < p - 1)
    //      {
    //        int send_rank = p * p - 1;
    //        MPI_Irecv(in_d, mesg_size, MPI_CHAR, send_rank,
    //            1, MPI_COMM_WORLD, &request[3]);
    ////        MPI_Irecv(in, 2, MPI_INT, send_rank,
    ////            1, MPI_COMM_WORLD, &request[3]);
    ////        printf("pi=%d, pj=%d, from=%d\n", pi, pj, send_rank);
    //        MPI_Wait(&request[3], &status[2]);
    //      }
    //    }
    //    else
    //    {
    //      int send_rank = pi * p + p - 1;
    //      MPI_Irecv(out_d, mesg_size, MPI_CHAR, send_rank,
    //          0, MPI_COMM_WORLD, &request[4]);
    //
    //      MPI_Wait(&request[4], &status[0]);
    //
    //      if (pi != pj)
    //      {
    //        int send_rank = pj * p + p - 1;
    //        MPI_Irecv(in_d, mesg_size, MPI_CHAR, send_rank,
    //            1, MPI_COMM_WORLD, &request[5]);
    ////        MPI_Irecv(in, 2, MPI_INT, send_rank,
    ////            1, MPI_COMM_WORLD, &request[5]);
    ////        printf("pi=%d, pj=%d, from=%d\n", pi, pj, send_rank);
    //        MPI_Wait(&request[5], &status[1]);
    //      }
    //    }
    //    if (pi == pj)
    //      cudaMemcpy(in_d, out_d, mesg_size, cudaMemcpyDeviceToDevice);
  }

};

#endif /* WAVE_H_ */
