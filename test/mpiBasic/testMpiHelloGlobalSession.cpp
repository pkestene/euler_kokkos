/**
 * \file testMpiHelloGlobalSession.cpp
 * \brief Mpi helloworld application using CPP wrapper classes from libHydroMpi.
 *
 *
 * To launch: mpirun -n 4 ./testMpiHelloGlobalSession
 *
 * Uses the MpiGlobalSession class to handle Initialize/Finalize operation.
 *
 * \date 1 Oct 2010
 * \author Pierre Kestener
 */
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <ctime>

#include <GlobalMpiSession.h>

int main(int argc,char **argv)
{
  int myRank, numProcs, namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME+1];

  time_t         curtime;
  struct tm     *loctime;

  // MPI resources
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Get_processor_name(processor_name,&namelength);
  
  // print local time on rank 0 machine
  if ( myRank == 0 ) {
    curtime = time (NULL);
    loctime = localtime (&curtime);
    std::cout << "Local time of process 0 : " << asctime (loctime) << std::endl;
  }
  
  // // print process rank and hostname
  std::cout << "MPI process " << myRank << " of " << numProcs << " is on " <<
    processor_name << std::endl;

  return EXIT_SUCCESS;
}
