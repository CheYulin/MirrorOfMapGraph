/* Copyright (C) 2009-2010 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <stdio.h>
#include <assert.h>
#define xfree free
#include <mpi.h>
#include "make_graph.h"

int main(int argc, char* argv[])
{
  int log_numverts;
  int size, rank;
  unsigned long my_edges;
  unsigned long global_edges;
  double start, stop;
  size_t i;

  MPI_Init(&argc, &argv);

  log_numverts = 16; /* In base 2 */
  long int edges_per_vert = 16;
  if (argc >= 2) log_numverts = atoi(argv[1]);
  if (argc >= 3) edges_per_vert = atoi(argv[2]);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) fprintf(stderr, "Graph size is %" PRId64 " vertices and %" PRId64 " edges\n", INT64_C(1) << log_numverts, edges_per_vert << log_numverts);

  /* Start of graph generation timing */
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  int64_t nedges;
  srand(128);
  packed_edge* result;
  make_graph(log_numverts, edges_per_vert << log_numverts, rand(), 8, &nedges, &result);
  MPI_Barrier(MPI_COMM_WORLD);
  stop = MPI_Wtime();
  /* End of graph generation timing */

  my_edges = nedges;

  for (i = 0; i < my_edges; ++i)
  {
    assert((get_v0_from_edge(&result[i]) >> log_numverts) == 0);
    assert((get_v1_from_edge(&result[i]) >> log_numverts) == 0);
  }

  MPI_Reduce(&my_edges, &global_edges, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0)
  {
    fprintf(stderr, "%lu edge%s generated in %fs (%f Medges/s on %d processor%s)\n", global_edges, (global_edges == 1 ? "" : "s"), (stop - start), global_edges / (stop - start) * 1.e-6, size, (size == 1 ? "" : "s"));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  if (rank == 0)
    fprintf(stderr, "\nFile IO\n");
  //MPI_Status status;
  MPI_File cFile[size];

  int length[size], prefix[size];
  char filename[size][10];

  //init first one to zero
  if (rank == 0)
    for (int i = 0; i < size; i++)
      length[i] = 0;

  //compute filenames
  for (int i = 0; i < size; i++)
    sprintf(filename[i], "graph%d", i);

  int p = sqrt(size);
  int global_size = INT64_C(1) << log_numverts;
  int slice_size = ceil((double)global_size / p);
  int x_index, y_index;


  if (rank == 0) printf("\nslice_size %d global_size:%d\n", slice_size, global_size);

  MPI_File sizefile;
  int rc = MPI_File_open(MPI_COMM_WORLD, "Graphsizes", MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &sizefile);
  if (rc)
  {
    printf("Unable to open file \"temp\"\n");
    fflush(stdout);
  }
  //open all files
  for (int i = 0; i < size; i++)
  {
    int rc = MPI_File_open(MPI_COMM_WORLD, filename[i], MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &cFile[i]);
    if (rc)
    {
      printf("Unable to open file \"temp\"\n");
      fflush(stdout);
    }
  }


  int from, to;
  int edge[2], file_index;
  if (rank == 0)
  {
    for (i = 0; i < size; i++)
      prefix[i] = length[i];

    //go through all elemnts
    for (i = 0; i < my_edges; ++i)
    {
      from = get_v0_from_edge(&result[i]);
      to = get_v1_from_edge(&result[i]);

      edge[0] = from % slice_size;
      edge[1] = to % slice_size;

      x_index = from / slice_size;
      y_index = to / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;

        //match elemnent found 

        //increment count for corresponding

        length[file_index]++;
      } //if(file_index == 0  ){printf("%d %d \n",edge[0],edge[1]);}

      edge[0] = to % slice_size;
      edge[1] = from % slice_size;

      x_index = to / slice_size;
      y_index = from / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;
        length[file_index]++;
      }


    }
    //send the sizes array to next node
    if (rank != size - 1) MPI_Send(&length, size, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);

    if (size == 1)
    {
      int writelength[size + 1];
      for (int i = 0; i < size; i++)
        writelength[i] = length[i];
      writelength[size] = (INT64_C(1) << log_numverts);

      printf("\nnumverts is %d\n", writelength[size]);

      MPI_File_write(sizefile, &writelength, size + 1, MPI_INT, MPI_STATUS_IGNORE);
    }


  }
  else if (rank < size - 1)
  {
    MPI_Recv(&length, size, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (i = 0; i < size; i++)
      prefix[i] = length[i];

    printf("\nIn Rank %d\n", rank);

    for (i = 0; i < my_edges; ++i)
    {
      from = get_v0_from_edge(&result[i]);
      to = get_v1_from_edge(&result[i]);

      edge[0] = from % slice_size;
      edge[1] = to % slice_size;

      x_index = from / slice_size;
      y_index = to / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;
        //if(file_index == 0  ){printf("%d %d \n",edge[0],edge[1]);}
        //match elemnent found 

        //increment count for corresponding

        length[file_index]++;
      } //write to corresponfing file
      //MPI_File_write(cFile[file_index], &edge, 2, MPI_INT, MPI_STATUS_IGNORE);

      edge[0] = to % slice_size;
      edge[1] = from % slice_size;

      x_index = to / slice_size;
      y_index = from / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;
        length[file_index]++;
      }


    }
    //send the sizes array to next node
    if (rank != size - 1) MPI_Send(&length, size, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);

  }
  else
  {

    MPI_Recv(&length, size, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (i = 0; i < size; i++)
      prefix[i] = length[i];
    printf("\nIn Rank %d\n", rank);
    for (i = 0; i < my_edges; ++i)
    {
      from = get_v0_from_edge(&result[i]);
      to = get_v1_from_edge(&result[i]);

      edge[0] = from % slice_size;
      edge[1] = to % slice_size;

      x_index = from / slice_size;
      y_index = to / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;
        //if(file_index == 0  ){printf("%d %d \n",edge[0],edge[1]);}
        //match elemnent found 

        //increment count for corresponding

        length[file_index]++;
      } //write to corresponfing file
      // MPI_File_write(cFile[file_index], &edge, 2, MPI_INT, MPI_STATUS_IGNORE);

      edge[0] = to % slice_size;
      edge[1] = from % slice_size;

      x_index = to / slice_size;
      y_index = from / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;
        length[file_index]++;
      }

    }

    int writelength[size + 1];
    for (int i = 0; i < size; i++)
      writelength[i] = length[i];
    writelength[size] = (INT64_C(1) << log_numverts);

    printf("\nnumverts is %d\n", writelength[size]);

    MPI_File_write(sizefile, &writelength, size + 1, MPI_INT, MPI_STATUS_IGNORE);


  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&sizefile);




  //create list

  int *dup_list[size];
  int count[size];
  for (i = 0; i < size; i++)
    count[i] = 0;



  for (int i = 0; i < size; i++)
    dup_list[i] = (int *)malloc(2 * sizeof (int)*(length[i] - prefix[i]));

  if (rank == 0)
  {
    //set view
    for (i = 0; i < my_edges; ++i)
    {
      from = get_v0_from_edge(&result[i]);
      to = get_v1_from_edge(&result[i]);

      edge[0] = from % slice_size;
      edge[1] = to % slice_size;

      x_index = from / slice_size;
      y_index = to / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;

        //match elemnent found 

        //if(file_index == 0  ){printf("%d %d \n",edge[0],edge[1]);}
        //write to dup_list
        dup_list[file_index][count[file_index]] = edge[0];
        dup_list[file_index][count[file_index] + 1] = edge[1];
        count[file_index] += 2;
      }

      edge[0] = to % slice_size;
      edge[1] = from % slice_size;

      x_index = to / slice_size;
      y_index = from / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;

        dup_list[file_index][count[file_index]] = edge[0];
        dup_list[file_index][count[file_index] + 1] = edge[1];
        count[file_index] += 2;
      }


    }
  }
  else if (rank < size - 1)
  {
    //set view
    for (i = 0; i < my_edges; ++i)
    {
      from = get_v0_from_edge(&result[i]);
      to = get_v1_from_edge(&result[i]);

      edge[0] = from % slice_size;
      edge[1] = to % slice_size;

      x_index = from / slice_size;
      y_index = to / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;

        //match elemnent found 

        //if(file_index == 1  ){printf("%d %d \n",edge[0],edge[1]);}

        dup_list[file_index][count[file_index]] = edge[0];
        dup_list[file_index][count[file_index] + 1] = edge[1];
        count[file_index] += 2;
      }

      edge[0] = to % slice_size;
      edge[1] = from % slice_size;

      x_index = to / slice_size;
      y_index = from / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;

        dup_list[file_index][count[file_index]] = edge[0];
        dup_list[file_index][count[file_index] + 1] = edge[1];
        count[file_index] += 2;
      }


    }

  }
  else
  {
    //set view


    for (i = 0; i < my_edges; ++i)
    {
      from = get_v0_from_edge(&result[i]);
      to = get_v1_from_edge(&result[i]);

      edge[0] = from % slice_size;
      edge[1] = to % slice_size;

      x_index = from / slice_size;
      y_index = to / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;

        //match elemnent found 

        //if(file_index == 0  ){printf("%d %d \n",edge[0],edge[1]);}
        dup_list[file_index][count[file_index]] = edge[0];
        dup_list[file_index][count[file_index] + 1] = edge[1];
        count[file_index] += 2;
      }

      edge[0] = to % slice_size;
      edge[1] = from % slice_size;

      x_index = to / slice_size;
      y_index = from / slice_size;
      if (x_index < p && y_index < p)
      {
        file_index = y_index * p + x_index;

        dup_list[file_index][count[file_index]] = edge[0];
        dup_list[file_index][count[file_index] + 1] = edge[1];
        count[file_index] += 2;
      }

    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  //if(rank ==0) for(int i=0;i<size;i++) printf("\n%d %d\n",i,count[i]);

  //NOW START WRITING YOUR PARTS

  if (rank == 0)
  {
    //set view	
    for (int i = 0; i < size; i++)
      MPI_File_write(cFile[i], dup_list[i], count[i], MPI_INT, MPI_STATUS_IGNORE);


  }
  else if (rank < size - 1)
  {
    //set view

    for (int i = 0; i < size; i++)
      MPI_File_seek(cFile[i], 2 * prefix[i] * sizeof (int), MPI_SEEK_SET);


    for (int i = 0; i < size; i++)
      MPI_File_write(cFile[i], dup_list[i], count[i], MPI_INT, MPI_STATUS_IGNORE);

  }
  else
  {
    //set view
    for (int i = 0; i < size; i++)
      MPI_File_seek(cFile[i], 2 * prefix[i] * sizeof (int), MPI_SEEK_SET);

    for (int i = 0; i < size; i++)
      MPI_File_write(cFile[i], dup_list[i], count[i], MPI_INT, MPI_STATUS_IGNORE);

  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (i = 0; i < size; i++)
  {
    MPI_File_close(&cFile[i]);
  }


  for (i = 0; i < size; i++)
    free(dup_list[i]);



  stop = MPI_Wtime();
  if (rank == 0) printf("\nTime taken to write the graph to files is %lf\n", stop - start);
  xfree(result);
  MPI_Finalize();
  return 0;
}
