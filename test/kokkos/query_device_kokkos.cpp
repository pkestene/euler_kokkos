/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <iostream>
#include <sstream>

#include <Kokkos_Macros.hpp>

#if defined( USE_MPI )
#include <mpi.h>
#endif // USE_MPI

#include <Kokkos_Core.hpp>

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

int main( int argc , char ** argv )
{

  UNUSED(argc);
  UNUSED(argv);

  std::ostringstream msg ;

  int mpi_rank = 0 ;
  int nRanks = 1;

  // just to avoid warning when built without MPI
  UNUSED(mpi_rank);
  UNUSED(nRanks);

#if defined( USE_MPI )

  MPI_Init( & argc , & argv );

  MPI_Comm_rank( MPI_COMM_WORLD , & mpi_rank );
  MPI_Comm_size( MPI_COMM_WORLD , & nRanks );

  msg << "MPI rank(" << mpi_rank << ") " ;

#endif // USE_MPI

  Kokkos::initialize(argc, argv);

#ifdef KOKKOS_ENABLE_CUDA
  {

    // // get device count
    // int devCount;
    // cudaGetDeviceCount(&devCount);

    // int devId = mpi_rank % devCount;
    // cudaSetDevice(devId);

    // To enable kokkos accessing multiple GPUs don't forget to
    // add option "--ndevices=X" where X is the number of GPUs
    // you want to use per node.

    // on a large cluster, the scheduler should assign ressources
    // in a way that each MPI task is mapped to a different GPU
    // let's cross-checked that:

    int cudaDeviceId;
    cudaGetDevice(&cudaDeviceId);
    std::cout << "I'm MPI task #" << mpi_rank << " (out of " << nRanks << ")"
	      << " pinned to GPU #" << cudaDeviceId << "\n";

  }
#endif // KOKKOS_ENABLE_CUDA

  msg << "{" << std::endl ;

  if ( Kokkos::hwloc::available() ) {
    msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
        << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
        << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
        << "] )"
        << std::endl ;
  }

  Kokkos::print_configuration( msg );

  msg << "}" << std::endl ;

  std::cout << msg.str();

  Kokkos::finalize();

#if defined( USE_MPI )

  MPI_Finalize();

#endif

  return 0 ;
}
