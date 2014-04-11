/*
 * wave.h
 *
 *  Created on: Apr 4, 2014
 *      Author: zhisong
 */
#include "mpi.h"
#include "kernel.cuh"
#ifndef WAVE_H_
#define WAVE_H_
using namespace std;
using namespace MPI;
using namespace mpikernel;
class wave
//frontier contraction in a 2-d partitioned graph
{
	int pi;	 //row
	int pj;  //column
	int p;
	int n;


public: wave(int l_pi,int l_pj,int l_p, int l_n) 
//l_pi is the x index
//l_pj is the y index
//l_p  is the number of partitions in 1d. usually, sqrt(number of processors)
//l_n  is the size of the problem, number of vertices
	{
		pi = l_pi;
		pj = l_pj;
		p  = l_p;
		n  = l_n;
	}

void propogate(char* out_d, char* assigned_d, char* prefix_d )
	//wave propogation, in sequential from top to bottom of the column
	{
	int p2 = sqrt(p);	
	unsigned int mesg_size = n/p2;
	int myid = pi*p2+pj;
	//int lastid = pi*p+p-1;
	int numthreads = 512;
	int numblocks = min(512,(int) ceil( n/p2) );
	MPI_Request request[2];
	MPI_Status  status[2];
	//if first one in the column, initiate the wave propogation
		if(pj == 0)
		{
			char *out_h = (char*)malloc(mesg_size);
			cudaMemcpy(out_h,out_d,mesg_size,cudaMemcpyDeviceToHost);
			MPI_Isend(out_h,mesg_size,MPI_CHAR,myid+1,pi,MPI_COMM_WORLD,&request[1]);
			MPI_Wait(&request[1],&status[1]);			
			free(out_h);
		}
	//else if not the last one, receive bitmap from top, process and send to next one	
	else if(pj != p2-1)
		{
			char *prefix_h = (char*)malloc(mesg_size);
			MPI_Irecv(prefix_h,mesg_size,MPI_CHAR,myid-1,pi,MPI_COMM_WORLD,&request[0] );
			MPI_Wait(&request[0],&status[0]);			
			cudaMemcpy(prefix_d,prefix_h,mesg_size,cudaMemcpyHostToDevice);
			mpikernel::bitsubstract<<<numblocks,numthreads>>>(n, out_d, prefix_d, assigned_d);				
			cudaDeviceSynchronize();
			mpikernel::bitunion<<<numblocks,numthreads>>>(n,out_d ,prefix_d, out_d);	
			char *out_h = (char*)malloc(mesg_size);
			cudaDeviceSynchronize();
			cudaMemcpy(out_h,out_d,mesg_size,cudaMemcpyDeviceToHost);
			MPI_Isend(out_h,mesg_size,MPI_CHAR,myid+1,pi,MPI_COMM_WORLD,&request[1]);
			free(prefix_h);
			MPI_Wait(&request[1],&status[1]);			
			free(out_h);
		}
	//else receive from the previous and then broadcast to the broadcast group 
	else 
		{
			char *prefix_h = (char*)malloc(mesg_size);
			MPI_Irecv(prefix_h,mesg_size,MPI_CHAR,myid-1,pi,MPI_COMM_WORLD,&request[0] );
			MPI_Wait(&request[0],&status[0]);			
			cudaMemcpy(prefix_d,prefix_h,mesg_size,cudaMemcpyHostToDevice);
			mpikernel::bitsubstract<<<numblocks,numthreads>>>(n, out_d, prefix_d, assigned_d);
			cudaDeviceSynchronize();
			mpikernel::bitunion<<<numblocks,numthreads>>>(n,out_d ,prefix_d, out_d);         
			cudaDeviceSynchronize();										          

		}

	}

};


#endif /* WAVE_H_ */
