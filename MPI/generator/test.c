#include<generator_mpi_api.h>
#include <stdlib.h>
#include <stdio.h>
#include<mpi.h>

int main(int argc, char** argv)
{
int numprocs, rank;
int ppr = 4; //partitions per rank
struct partition partitions[ppr];
int i,j;
 MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &numprocs); 
  rank = atoi(getenv("MV2_COMM_WORLD_RANK"));

  gen_2d(26, 16, ppr, partitions);


//for(j=0;j<ppr;j++)
//for(i=0;i<partitions[j].numedges;i++)
//	printf("%ld %ld \n",partitions[j].edges[2*i],partitions[j].edges[2*i+1] );

return 0;

}
