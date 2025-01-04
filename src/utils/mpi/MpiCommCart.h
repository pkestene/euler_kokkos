/**
 * \file MpiCommCart.h
 * \brief A simple derive class of MpiComm to handle cartesian
 * topology.
 *
 */
#ifndef MPI_COMM_CART_H_
#define MPI_COMM_CART_H_

#include "MpiComm.h"
#include "mpiEnums.h"

namespace euler_kokkos
{

/**
 * \brief Object representation of an MPI communicator with
 * cartesian virtual topology.
 */
class MpiCommCart : public MpiComm
{
public:
  //! Construct a MpiCommCart using a 2D cartesian virtual topology
  MpiCommCart(int mx, int my, int isPeriodic, int allowReorder);

  //! Construct a MpiCommCart using a 3D cartesian virtual topology
  MpiCommCart(int mx, int my, int mz, int isPeriodic, int allowReorder);

  //! Destructor
  virtual ~MpiCommCart();

private:
  int   mx_, my_, mz_;
  int * myCoords_;
  bool  is2D;

public:
  //! get topology dimension (2 or 3) from MPI_Cartdim_get just for cross-checking
  virtual int
  getDim() const;

  //! use MPI_Cart_rank to get rank of the shifted process
  virtual int
  getCartRank(const int coords[]) const;

  //! return cartesian coordinate of a process of a given rank
  virtual void
  getCoords(int rank, int maxdims, int coords[]) const;

  //! return my coordinates
  virtual void
  getMyCoords(int coords[]);

  //! use MPI_Cart_shift to get rank of the shifted processes
  //! regarding the given direction
  virtual void
  shift(int direction, int disp, int & rank_source, int & rank_dest) const;

  //! return rank of the neighbor process identified by the template parameter
  template <NeighborLocation nl>
  int
  getNeighborRank() const;
};

// =======================================================
// =======================================================
inline int
MpiCommCart::getDim() const
{
  int ndims;
  CHECK_MPI_ERR(::MPI_Cartdim_get(this->get_MPI_Comm(), &ndims));
  return ndims;
}

// =======================================================
// =======================================================
inline int
MpiCommCart::getCartRank(const int coords[]) const
{
  int rank;
  CHECK_MPI_ERR(::MPI_Cart_rank(this->get_MPI_Comm(), const_cast<int *>(coords), &rank));
  return rank;
}

// =======================================================
// =======================================================
inline void
MpiCommCart::getCoords(int rank, int maxdims, int coords[]) const
{
  CHECK_MPI_ERR(::MPI_Cart_coords(this->get_MPI_Comm(), rank, maxdims, coords));
}

// =======================================================
// =======================================================
inline void
MpiCommCart::getMyCoords(int coords[])
{
  int maxDim = (is2D ? 2 : 3);
  for (int i = 0; i < maxDim; ++i)
  {
    coords[i] = myCoords_[i];
  }
}

// =======================================================
// =======================================================
inline void
MpiCommCart::shift(int direction, int disp, int & rank_source, int & rank_dest) const
{
  CHECK_MPI_ERR(::MPI_Cart_shift(this->get_MPI_Comm(), direction, disp, &rank_source, &rank_dest));
}

// =======================================================
// =======================================================
template <NeighborLocation nl>
int
MpiCommCart::getNeighborRank() const
{

  int rank;

  if (is2D)
  { // 2D
    int coords[NDIM_2D];
    coords[0] = myCoords_[0];
    coords[1] = myCoords_[1];

    if (nl == X_MIN)
    {
      coords[0] -= 1;
    }
    else if (nl == X_MAX)
    {
      coords[0] += 1;
    }
    else if (nl == Y_MIN)
    {
      coords[1] -= 1;
    }
    else if (nl == Y_MAX)
    {
      coords[1] += 1;
    }
    rank = getCartRank(coords);
  }
  else
  { // 3D
    int coords[NDIM_3D];
    coords[0] = myCoords_[0];
    coords[1] = myCoords_[1];
    coords[2] = myCoords_[2];

    if (nl == X_MIN)
    {
      coords[0] -= 1;
    }
    else if (nl == X_MAX)
    {
      coords[0] += 1;
    }
    else if (nl == Y_MIN)
    {
      coords[1] -= 1;
    }
    else if (nl == Y_MAX)
    {
      coords[1] += 1;
    }
    else if (nl == Z_MIN)
    {
      coords[2] -= 1;
    }
    else if (nl == Z_MAX)
    {
      coords[2] += 1;
    }
    rank = getCartRank(coords);
  }

  return rank;
}

} // namespace euler_kokkos

#endif // MPI_COMM_CART_H_
