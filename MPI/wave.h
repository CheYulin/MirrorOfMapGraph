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

void propogate(char* out, char* assigned, char* prefix )
	//wave propogation, in sequential from top to bottom of the column
	{
	unsigned int mesg_size = n/(sizeof(char)*p);
	int myid = pi*p+pj;
	//int lastid = pi*p+p-1;
	MPI_Request request[2];
	MPI_Status  status[2];
	//if first one in the column, initiate the wave propogation
		if(pj == 0)
		{
			MPI_Isend(out,mesg_size,MPI_CHAR,myid+1,pi,MPI_COMM_WORLD,&request[1]);
			MPI_Wait(&request[1],&status[1]);			
//			MPI_Irecv(out,mesg_size,MPI_CHAR,lastid,pi,MPI_COMM_WORLD,request[0]);
		}
	//else if not the last one, receive bitmap from top, process and send to next one	
	else if(pj != p-1)
		{
			MPI_Irecv(prefix,mesg_size,MPI_CHAR,myid-1,pi,MPI_COMM_WORLD,&request[0] );
			mpikernel::bitsubstract<<<512,512>>>(n, out, prefix, assigned);	
			cudaDeviceSynchronize();
			mpikernel::bitunion<<<512,512>>>(n,out ,prefix, out);	
			cudaDeviceSynchronize();
			MPI_Isend(out,mesg_size,MPI_CHAR,myid+1,pi,MPI_COMM_WORLD,&request[1]);
			MPI_Wait(&request[1],&status[1]);			
		}
	//else receive from the previous and then broadcast to the broadcast group 
	else 
		{
      MPI_Irecv(prefix,mesg_size,MPI_CHAR,myid-1,pi,MPI_COMM_WORLD,&request[0] );
      mpikernel::bitsubstract<<<512,512>>>(n, out, prefix, assigned);
			cudaDeviceSynchronize();
			mpikernel::bitunion<<<512,512>>>(n,out ,prefix, out);         
			cudaDeviceSynchronize();
												          
//			MPI_Bcast();
		}
	}

};


#endif /* WAVE_H_ */
