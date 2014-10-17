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
#include <string.h>
#define COMPACT_FACTOR 1000
#define xfree free
#include <mpi.h>
#include "make_graph.h"
#include "generator_mpi_api.h"

int gen_2d(int log_numverts, int edges_per_vert, int ppr, struct partition* parts)
{
	int size, rank;
	unsigned long my_edges;
	unsigned long global_edges;
	double start, stop;
	size_t i;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//if (rank == 0) fprintf(stderr, "Graph size is %" PRId64 " vertices and %" PRId64 " edges as %d partitions\n", INT64_C(1) << log_numverts, edges_per_vert << log_numverts, ppr*size);

	/* Start of graph generation timing */
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	int64_t nedges;
	srand(128+rank);
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

	
	unsigned long *length = calloc(size*ppr,sizeof(unsigned long));
	unsigned long *recv_length = calloc(size*ppr,sizeof(unsigned long));

	int p = sqrt(size);
	unsigned long global_size = INT64_C(1) << log_numverts;
	unsigned long slice_size = ceil((long double)global_size / p);
	unsigned long slice_rank = ceil((long double)slice_size / ppr);
	int x_index, y_index;

	if (rank == 0) printf("\nslice_size %ld Sub Slice Size %ld  global_size:%ld\n", slice_size, slice_rank, global_size);

	unsigned long from, to;
	int file_index,file_sub_index;
	unsigned long edge[2];

		
	//go through all elemnts
	for (i = 0; i < my_edges; ++i)
	{
		from = get_v0_from_edge(&result[i]);
		to = get_v1_from_edge(&result[i]);

		edge[0] = from % slice_size;
		edge[1] = to % slice_size;

		x_index = (from / slice_size);
		y_index = (to / slice_size);
		if (x_index < p && y_index < p)
		{
			file_index = y_index * p + x_index;
			file_sub_index = edge[1] / slice_rank;

			length[file_index*ppr+file_sub_index]+=1;
		} //if(file_index == 0  ){printf("%d %d \n",edge[0],edge[1]);}

		edge[0] = to % slice_size;
		edge[1] = from % slice_size;

		x_index = to / slice_size;
		y_index = from / slice_size;
		if (x_index < p && y_index < p)
		{
			file_index = y_index * p + x_index;
			file_sub_index = edge[1] / slice_rank;

			length[file_index*ppr+file_sub_index]++;
		}
	}


	//we now have the counts to send in *length
	//do an mpi alltoall to share what you are sending to other processes
	MPI_Alltoall(length, ppr, MPI_UNSIGNED_LONG, recv_length, ppr, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

	unsigned long *send_prefix = malloc(sizeof(unsigned long)*size*ppr); 
	unsigned long *send_sizes = malloc(sizeof(unsigned long)*size);
	int *sdspls = malloc(sizeof(int)*(size+1)); 
	int *send_sizes_int = malloc(sizeof(int)*(size)); 

	send_prefix[0]=0;
	send_sizes[0] = length[0]*2;
	sdspls[0]=0;

	for(int j=1;j<ppr;j++)
	{
		send_prefix[j] = send_prefix[j-1] + length[j-1]*2;
		send_sizes[0] += length[j]*2;
	}
	send_sizes_int[0] = (send_sizes[0]+COMPACT_FACTOR)/COMPACT_FACTOR;

	for(int i=1;i<size;i++)
	{
		send_prefix[i*ppr]=send_prefix[(i-1)*ppr]+send_sizes_int[i-1] * COMPACT_FACTOR ;
		send_sizes[i] = length[i*ppr]*2;
		for(int j=1;j<ppr;j++)
		{
			send_prefix[i*ppr+j] = send_prefix[i*ppr+j-1] + length[i*ppr+j-1]*2;
			send_sizes[i] += length[i*ppr+j]*2;
		}
		send_sizes_int[i]=(send_sizes[i]+COMPACT_FACTOR)/COMPACT_FACTOR;
		sdspls[i] = sdspls[i-1] + send_sizes_int[i-1];
	}
	sdspls[size] = sdspls[size-1] + send_sizes_int[size-1];


	unsigned long *recv_prefix = calloc(size*ppr, sizeof(unsigned long)); 
	unsigned long *rsize = malloc(sizeof(unsigned long)*(size));
	unsigned long *part_count = calloc(ppr,sizeof(unsigned long));
	int *rdispls = malloc(sizeof(int)*(size+1));
	int *rsize_int = malloc(sizeof(int)*size);

	recv_prefix[0]=0;
	rsize[0] = recv_length[0]*2;
	rdispls[0]=0;
	part_count[0] += recv_length[0];

	for(int j=1;j<ppr;j++)
	{
		recv_prefix[j] = recv_prefix[j-1] + recv_length[j-1]*2;
		rsize[0] += recv_length[j]*2;
		part_count[j] += recv_length[j];
	}
	rsize_int[0] = (rsize[0]+COMPACT_FACTOR)/COMPACT_FACTOR;
	
	for(int i=1;i<size;i++)
	{
		recv_prefix[i*ppr]=recv_prefix[(i-1)*ppr]+rsize_int[i-1] * COMPACT_FACTOR ;
		rsize[i] = recv_length[i*ppr]*2;
		part_count[0] += recv_length[(i)*ppr];
		for(int j=1;j<ppr;j++)
		{
			recv_prefix[i*ppr+j] = recv_prefix[i*ppr+j-1] + recv_length[i*ppr+j-1]*2;
			rsize[i] += recv_length[i*ppr+j]*2;
			part_count[j] += recv_length[(i)*ppr+j];
		}
		rsize_int[i]=(rsize[i]+COMPACT_FACTOR)/COMPACT_FACTOR;
		rdispls[i] = rdispls[i-1] + rsize_int[i-1];
	}
	rdispls[size] = rdispls[size-1] + rsize_int[size-1];

	//create list

	unsigned long *dup_list = malloc(sizeof(unsigned long)*sdspls[size]*COMPACT_FACTOR);
	//go through all elemnts and add to list
	for (i = 0; i < my_edges; ++i)
	{
		from = get_v0_from_edge(&result[i]);
		to = get_v1_from_edge(&result[i]);

		edge[0] = from % slice_size;
		edge[1] = to % slice_size;

		x_index = from / slice_size;
		y_index = to / slice_size;

		//printf("%ld %ld\n",edge[0],edge[1]);
		//printf("%ld %ld\n",edge[1],edge[0]);


		if (x_index < p && y_index < p)
		{
			file_index = y_index * p + x_index;
			file_sub_index = edge[1] / slice_rank;

			dup_list[send_prefix[file_index*ppr+file_sub_index]] = edge[0];
			dup_list[send_prefix[file_index*ppr+file_sub_index]+1] = edge[1];

			send_prefix[file_index*ppr+file_sub_index]+=2;
		} //if(file_index == 0  ){printf("%d %d \n",edge[0],edge[1]);}

		edge[0] = to % slice_size;
		edge[1] = from % slice_size;

		x_index = to / slice_size;
		y_index = from / slice_size;
		if (x_index < p && y_index < p)
		{
			file_index = y_index * p + x_index;
			file_sub_index = edge[1] / slice_rank;
	

			

			dup_list[send_prefix[file_index*ppr+file_sub_index]] = edge[0];
			dup_list[send_prefix[file_index*ppr+file_sub_index]+1] = edge[1];

			send_prefix[file_index*ppr+file_sub_index]+=2;
			
		}
	}

	unsigned long *edges_owned = malloc(sizeof(unsigned long)*rdispls[size]*COMPACT_FACTOR);

	MPI_Datatype packed_type;
	MPI_Type_contiguous(COMPACT_FACTOR, MPI_UNSIGNED_LONG, &packed_type);

	MPI_Type_commit(&packed_type);
	
	MPI_Alltoallv(dup_list, send_sizes_int, sdspls, packed_type, edges_owned, rsize_int, rdispls, packed_type, MPI_COMM_WORLD);


	free(sdspls);
	free(send_sizes);
	free(send_sizes_int);
	free(send_prefix);
	

	free(rsize);
	free(rdispls);
	free(rsize_int);

	free(dup_list);
	free(length);


	for(int i=0;i<ppr;i++)
	{
		parts[i].numedges = part_count[i];
		parts[i].edges = malloc(sizeof(unsigned long)*part_count[i]*2);
	}


	memset(part_count,0,ppr*(sizeof(unsigned long)));

	for(int i=0;i<size;i++)
		{
			for(int j=0;j<ppr;j++)
			{
	
				memcpy(&parts[j].edges[part_count[j]], &edges_owned[recv_prefix[i*ppr+j]], sizeof(unsigned long)*recv_length[i*ppr+j]*2);
				part_count[j] += recv_length[i*ppr+j]*2;
			}

		}

	free(edges_owned);
	free(recv_prefix);
	free(part_count);
	free(recv_length);

	stop = MPI_Wtime();
	if(rank == 0)
		printf("Edges partitioned in %lfs\n",stop-start);
	MPI_Finalize();
	return 0;
}
