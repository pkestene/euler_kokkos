/**
 * \file testMpiHello.cpp
 * \brief Small MPI test using c++ API.
 *
 * To launch: mpirun -n 4 ./testMpiHello
 *
 * !!! WARNING C++ binding are deprecated in MPI 2.2 !!!
 *
 * \ingroup test
 *
 * \date 27 Sept 2010
 * \author Pierre Kestener
 */
// the following pragma does the same as -Wno-unused-parameter given
// in Makefile.am
//#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cstring>

int main(int argc,char **argv)
{
  int myRank, numProcs, namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME+1];

  time_t         curtime;
  struct tm     *loctime;

  // MPI initialize
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_size(MPI_COMM_WORLD, &myRank);
  MPI_Get_processor_name(processor_name, &namelength);
  
  // print local time on rank 0 machine
  if ( myRank == 0 ) {
    curtime = time (NULL);
    loctime = localtime (&curtime);
    std::cout << "Local time of process 0 : " << asctime (loctime) << std::endl;
  }
  
  // print process rank and hostname
  std::cout << "MPI process " << myRank << " of " << numProcs << " is on " <<
    processor_name << std::endl;
  
  // MPI finalize 
  MPI_Finalize();
  return EXIT_SUCCESS;
}
