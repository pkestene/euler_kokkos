// Example that shows simple usage of the INIReader class

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream> // string stream

#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>

#include "config/ConfigMap.h"

// =====================================================================
// =====================================================================
// =====================================================================
int test1(std::string filename) {

  int myRank;
  int nTasks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nTasks);

  
  char* buffer = nullptr;
  int buffer_size = 0;
  
  // MPI rank 0 reads parameter file
  if (myRank == 0) {

    std::cout << "Rank 0 reading input file....\n";
    
    // open file and go to the end to get file size in bytes
    std::ifstream filein(filename.c_str(), std::ifstream::ate);
    int file_size = filein.tellg();

    filein.seekg(0); // rewind
    
    buffer_size = file_size;
    buffer = new char[buffer_size];
    
    if(filein.read(buffer, buffer_size))
      std::cout << buffer << '\n';
    
  }

  // broacast buffer size (collective)
  MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (myRank>0) {
    printf("I'm rank %d allocating buffer of size %d\n",myRank,buffer_size);
    buffer = new char[buffer_size];
  }

  // broastcast buffer itself (collective)
  MPI_Bcast(&buffer[0], buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);


  // now all MPI rank should have buffer filled, try to build a ConfigMap
  ConfigMap configMap(buffer,buffer_size);

  if (buffer)
    delete [] buffer;

  // just for checking, choose one tank and print the configMap
  if (myRank==nTasks-1) {
    std::cout << configMap << std::endl;    
  }
  
  return 0;
  
} // test1()  

// =====================================================================
// =====================================================================
// =====================================================================
int main(int argc, char* argv[])
{

  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
  
  std::string input_file;
  
  if (argc>1) {
    input_file = std::string(argv[1]);  
  } else {
    input_file = "test_mpi.ini";
  }

  int status = test1(input_file);
  
  return status;
  
} // main
