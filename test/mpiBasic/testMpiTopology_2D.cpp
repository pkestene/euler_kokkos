/**
 * \file testMpiTopology_2D.cpp
 * \brief Simple example showing how to use MPI virtual topology (2D)
 *
 *
 * Simple MPI test with a cartesian grid topology.
 * In this example, a 3 by 4 grid is instantiate.
 *
 * !!! WARNING : C++ binding are deprecated in MPI 2.2 !!!
 *
 *
 * IMPORTANT NOTE.
 * The purpose of class MpiCommCart was to avoid the use of the
 * original C++ API which is deprecated in MPI 2.2 standards, but
 * the Teuchos API was far from being complete to be really useable
 * here; so we added some point-to-point communication routines.
 *
 * We already have introduced Cartesian topology inside this
 * framework.
 *
 */

#include <mpi.h>

#include <iostream>
#include <cstdlib>
#include <unistd.h> // for sleep

#include <ParallelEnv.h>
#include <MpiCommCart.h>

constexpr int SIZE_X = 2;
constexpr int SIZE_Y = 2;
constexpr int SIZE_Z = 4;
constexpr int SIZE_2D = (SIZE_X * SIZE_Y);
constexpr int SIZE_3D = (SIZE_X * SIZE_Y * SIZE_Z);

constexpr int N_NEIGHBORS_2D = 4;
constexpr int N_NEIGHBORS_3D = 6;

constexpr int NDIM = 2;

namespace euler_kokkos
{
// =====================================================================
// =====================================================================
void
test_cartesian_topology(ParallelEnv & par_env, int argc, char * argv[])
{
  auto worldComm = euler_kokkos::MpiComm();

  auto       myRank = worldComm.rank();
  const auto numTasks = worldComm.size();

  int  namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  CHECK_MPI_ERR(::MPI_Get_processor_name(processor_name, &namelength));

  // print warning
  if (myRank == 0)
  {
    std::cout << "Take care that MPI Cartesian Topology uses COLUMN MAJOR-FORMAT !!!\n";
    std::cout << "\n";
    std::cout << "In this test, each MPI process of the cartesian grid sends a message\n";
    std::cout << "containing a integer (rank of the current process) to all of its\n";
    std::cout << "neighbors. So you must check that arrays \"neighbors\" and \"inbuf\"\n";
    std::cout << "contain the same information !\n\n";
  }

  // 2D CARTESIAN MPI MESH
  if (numTasks == SIZE_2D)
  {
    int nbrs[N_NEIGHBORS_2D];
    int periods = euler_kokkos::MPI_CART_PERIODIC_TRUE;
    int reorder = euler_kokkos::MPI_REORDER_TRUE;
    int coords[NDIM];

    // create the cartesian topology
    par_env.setup_cartesian_topology(SIZE_X, SIZE_Y, (int)periods, (int)reorder);

    auto & cartcomm = dynamic_cast<MpiCommCart &>(par_env.comm());

    // get rank inside the topology
    myRank = cartcomm.rank();

    // get 2D coordinates inside topology
    cartcomm.getMyCoords(coords);

    // get rank of source (x-1) and destination (x+1) process
    // take care MPI uses column-major order
    // get rank of source (y-1) and destination (y+1) process
    nbrs[euler_kokkos::X_MIN] = cartcomm.getNeighborRank<euler_kokkos::X_MIN>();
    nbrs[euler_kokkos::X_MAX] = cartcomm.getNeighborRank<euler_kokkos::X_MAX>();
    nbrs[euler_kokkos::Y_MIN] = cartcomm.getNeighborRank<euler_kokkos::Y_MIN>();
    nbrs[euler_kokkos::Y_MAX] = cartcomm.getNeighborRank<euler_kokkos::Y_MAX>();

    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * N_NEIGHBORS_2D);

    int source, dest, i, tag = 1;

    auto inbuf = Kokkos::View<int *>("inbuf", N_NEIGHBORS_2D);
    Kokkos::deep_copy(inbuf, MPI_PROC_NULL);

    auto outbuf = Kokkos::View<int *>("outbuf", N_NEIGHBORS_2D);
    Kokkos::deep_copy(outbuf, myRank);

    // send    my rank to   each of my neighbors
    // receive my rank from each of my neighbors
    // inbuf should contain the rank of all neighbors
    for (i = 0; i < N_NEIGHBORS_2D; i++)
    {
      dest = nbrs[i];
      source = nbrs[i];
      auto recv_range = std::pair<std::size_t, std::size_t>(i, i + 1);

      auto outbuf_send = Kokkos::subview(outbuf, recv_range);
      reqs[i] = cartcomm.MPI_Isend(outbuf_send, dest, tag);

      auto inbuf_recv = Kokkos::subview(inbuf, recv_range);
      reqs[i + N_NEIGHBORS_2D] = cartcomm.MPI_Irecv(inbuf_recv, source, tag);
    }
    cartcomm.MPI_Waitall(2 * N_NEIGHBORS_2D, reqs.data());

    printf("rank= %2d coords= %d %d  neighbors(x-,+-,y-,y+) = %2d %2d %2d %2d\n",
           myRank,
           coords[0],
           coords[1],
           nbrs[euler_kokkos::X_MIN],
           nbrs[euler_kokkos::X_MAX],
           nbrs[euler_kokkos::Y_MIN],
           nbrs[euler_kokkos::Y_MAX]);
    printf("rank= %2d coords= %d %d  inbuf    (x-,x+,y-,y+) = %2d %2d %2d %2d\n",
           myRank,
           coords[0],
           coords[1],
           inbuf[euler_kokkos::X_MIN],
           inbuf[euler_kokkos::X_MAX],
           inbuf[euler_kokkos::Y_MIN],
           inbuf[euler_kokkos::Y_MAX]);

    // print topology
    cartcomm.MPI_Barrier();
    sleep(1);

    if (myRank == 0)
    {
      printf("Print topology (COLUMN MAJOR-ORDER) for %dx%d 2D grid:\n", SIZE_X, SIZE_Y);
      printf(" rank     i     j\n");
    }
    printf("%5d %5d %5d %10d %10d %10d %10d\n",
           myRank,
           coords[0],
           coords[1],
           nbrs[euler_kokkos::X_MIN],
           nbrs[euler_kokkos::X_MAX],
           nbrs[euler_kokkos::Y_MIN],
           nbrs[euler_kokkos::Y_MAX]);
  }
  else
  {
    std::cout << "Must specify " << SIZE_2D << " processors. Terminating.\n";
  }
} // test

} // namespace euler_kokkos

// =====================================================================
// =====================================================================
// =====================================================================
int
main(int argc, char * argv[])
{
  // MPI resources
  // auto mpiSession = euler_kokkos::GlobalMpiSession(argc, argv);
  auto par_env = euler_kokkos::ParallelEnv(argc, argv);

  euler_kokkos::test_cartesian_topology(par_env, argc, argv);

  return EXIT_SUCCESS;
}
