/**
 * \file MpiCommCart.cpp
 * \brief Implements class MpiCommCart
 *
 */

#include "MpiCommCart.h"

namespace euler_kokkos
{

// =======================================================
// =======================================================
MpiCommCart::MpiCommCart(int mx, int my, int isPeriodic, int allowReorder)
  : MpiComm()
  , mx_(mx)
  , my_(my)
  , mz_(0)
  , myCoords_(new int[NDIM_2D])
  , is2D(true)
{

  int dims[NDIM_2D] = { mx, my };
  int periods[NDIM_2D] = { isPeriodic, isPeriodic };

  // create virtual topology cartesian 2D
  MPI_Comm new_comm;
  CHECK_MPI_ERR(::MPI_Cart_create(MPI_COMM_WORLD, NDIM_2D, dims, periods, allowReorder, &new_comm));

  // take ownership
  this->comm_ptr.reset(new MPI_Comm(new_comm), comm_free());

  // get cartesian coordinates (myCoords_) of current process
  const auto my_rank = this->rank();
  getCoords(my_rank, NDIM_2D, myCoords_);
}

// =======================================================
// =======================================================
MpiCommCart::MpiCommCart(int mx, int my, int mz, int isPeriodic, int allowReorder)
  : MpiComm()
  , mx_(mx)
  , my_(my)
  , mz_(mz)
  , myCoords_(new int[NDIM_3D])
  , is2D(false)
{
  int dims[NDIM_3D] = { mx, my, mz };
  int periods[NDIM_3D] = { isPeriodic, isPeriodic, isPeriodic };

  // create virtual topology cartesian 3D
  MPI_Comm new_comm;
  CHECK_MPI_ERR(::MPI_Cart_create(MPI_COMM_WORLD, NDIM_3D, dims, periods, allowReorder, &new_comm));

  // take ownership
  this->comm_ptr.reset(new MPI_Comm(new_comm), comm_free());

  // get cartesian coordinates (myCoords_) of current process
  const auto my_rank = this->rank();
  getCoords(my_rank, NDIM_3D, myCoords_);
}

// =======================================================
// =======================================================
MpiCommCart::~MpiCommCart()
{
  delete[] myCoords_;
}

} // namespace euler_kokkos
