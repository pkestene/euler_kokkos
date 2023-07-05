/**
 * \file mpiBorderUtils.h
 * \brief Some utility routines dealing with MPI border buffers.
 *
 * \date 13 Oct 2010
 * \author Pierre Kestener
 *
 */
#ifndef MPI_BORDER_UTILS_H_
#define MPI_BORDER_UTILS_H_

#include "shared/kokkos_shared.h"
#include "shared/enums.h"

namespace euler_kokkos
{

/**
 * \class CopyBorderBuf_To_DataArray
 *
 * Copy a border buffer (as received by MPI communications) into the
 * right location (given by template parameter boundaryLoc).
 * Here we assume U is a DataArray.
 *
 * template parameters:
 * @tparam boundaryLoc : destination boundary location
 *                       used to check array dimensions and set offset
 * @tparam dimType     : triggers 2D or 3D specific treatment
 *
 * argument parameters:
 * @param[out] U reference to a hydro simulations array (destination array)
 * @param[in]  b reference to a border buffer (source array)
 * @param[in]  ghostWidth is the number of ghost cells
 *
 */
template <BoundaryLocation boundaryLoc, DimensionType dimType>
class CopyBorderBuf_To_DataArray
{

public:
  //! Decide at compile-time which data array to use
  using DataArray = typename std::conditional<dimType == TWO_D, DataArray2d, DataArray3d>::type;


  CopyBorderBuf_To_DataArray(DataArray U, DataArray b, int ghostWidth)
    : U(U)
    , b(b)
    , ghostWidth(ghostWidth){};

  // static method which does it all: create and execute functor
  static void
  apply(DataArray U, DataArray b, int ghostWidth, int nbIter)
  {
    CopyBorderBuf_To_DataArray<boundaryLoc, dimType> functor(U, b, ghostWidth);
    Kokkos::parallel_for(nbIter, functor);
  }


  template <DimensionType dimType_ = dimType>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dimType_ == TWO_D, int>::type & index) const
  {

    const int isize = U.extent(0);
    const int jsize = U.extent(1);
    const int nbvar = U.extent(2);
    int       i, j;

    /*
     * Proceed with copy.
     */
    int offset = 0;
    if (boundaryLoc == XMAX)
      offset = U.extent(0) - ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.extent(1) - ghostWidth;

    if (boundaryLoc == XMIN or boundaryLoc == XMAX)
    {

      // j = index / ghostWidth;
      // i = index - j*ghostWidth;
      index2coord(index, i, j, ghostWidth, jsize);

      for (int nVar = 0; nVar < nbvar; ++nVar)
        U(offset + i, j, nVar) = b(i, j, nVar);
    }
    else if (boundaryLoc == YMIN or boundaryLoc == YMAX)
    {

      // i = index / ghostWidth;
      // j = index - i*ghostWidth;
      index2coord(index, i, j, isize, ghostWidth);

      for (int nVar = 0; nVar < nbvar; ++nVar)
        U(i, offset + j, nVar) = b(i, j, nVar);
    }

  } // operator() - 2D

  template <DimensionType dimType_ = dimType>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dimType_ == THREE_D, int>::type & index) const
  {

    const int isize = U.extent(0);
    const int jsize = U.extent(1);
    const int ksize = U.extent(2);
    const int nbvar = U.extent(3);
    int       i, j, k;

    /*
     * Proceed with copy.
     */
    int offset = 0;
    if (boundaryLoc == XMAX)
      offset = U.extent(0) - ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.extent(1) - ghostWidth;
    if (boundaryLoc == ZMAX)
      offset = U.extent(2) - ghostWidth;


    if (boundaryLoc == XMIN or boundaryLoc == XMAX)
    {

      index2coord(index, i, j, k, ghostWidth, jsize, ksize);

      for (int nVar = 0; nVar < nbvar; ++nVar)
      {
        U(offset + i, j, k, nVar) = b(i, j, k, nVar);
      }
    }
    else if (boundaryLoc == YMIN or boundaryLoc == YMAX)
    {

      index2coord(index, i, j, k, isize, ghostWidth, ksize);

      for (int nVar = 0; nVar < nbvar; ++nVar)
      {
        U(i, offset + j, k, nVar) = b(i, j, k, nVar);
      }
    }
    else if (boundaryLoc == ZMIN or boundaryLoc == ZMAX)
    {

      index2coord(index, i, j, k, isize, jsize, ghostWidth);

      for (int nVar = 0; nVar < nbvar; ++nVar)
      {
        U(i, j, offset + k, nVar) = b(i, j, k, nVar);
      }
    }

  } // operator() - 3D

  DataArray U;
  DataArray b;
  int       ghostWidth;

}; // class CopyBorderBuf_To_DataArray

/**
 * \class CopyDataArray_To_BorderBuf
 *
 * Copy array border to a border buffer (to be sent by MPI communications)
 * Here we assume U is a <b>DataArray</b>.
 * \sa copyBorderBufSendToHostArray
 *
 * When used with 2d Array, we just use the index to access border data.
 * For 3d data Array, we need to map the correct location, by using index2coord.
 *
 * template parameters:
 * @tparam boundaryLoc : boundary location in source Array
 * @tparam dimType     : triggers 2D or 3D specific treatment
 *
 * argument parameters:
 * @param[out] b reference to a border buffer (destination array)
 * @param[in]  U reference to a hydro simulations array (source array)
 * @param[in]  ghostWidth is the number of ghost cells
 */
template <BoundaryLocation boundaryLoc, DimensionType dimType>
class CopyDataArray_To_BorderBuf
{

public:
  //! Decide at compile-time which data array to use
  using DataArray = typename std::conditional<dimType == TWO_D, DataArray2d, DataArray3d>::type;

  CopyDataArray_To_BorderBuf(DataArray b, DataArray U, int ghostWidth)
    : b(b)
    , U(U)
    , ghostWidth(ghostWidth){};

  // static method which does it all: create and execute functor
  static void
  apply(DataArray b, DataArray U, int ghostWidth, int nbIter)
  {
    CopyDataArray_To_BorderBuf<boundaryLoc, dimType> functor(b, U, ghostWidth);
    Kokkos::parallel_for(nbIter, functor);
  }

  template <DimensionType dimType_ = dimType>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dimType_ == TWO_D, int>::type & index) const
  {

    const int isize = U.extent(0);
    const int jsize = U.extent(1);
    const int nbvar = U.extent(2);
    int       i, j;

    /*
     * Proceed with copy
     */
    int offset = ghostWidth;
    if (boundaryLoc == XMAX)
      offset = U.extent(0) - 2 * ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.extent(1) - 2 * ghostWidth;

    /*
     * simple copy when PERIODIC or COPY
     */
    if (boundaryLoc == XMIN or boundaryLoc == XMAX)
    {

      // j = index / ghostWidth;
      // i = index - j*ghostWidth;
      index2coord(index, i, j, ghostWidth, jsize);

      for (int nVar = 0; nVar < nbvar; ++nVar)
      {
        b(i, j, nVar) = U(offset + i, j, nVar);
      }
    }
    else if (boundaryLoc == YMIN or boundaryLoc == YMAX)
    {

      // i = index / ghostWidth;
      // j = index - i*ghostWidth;
      index2coord(index, i, j, isize, ghostWidth);

      for (int nVar = 0; nVar < nbvar; ++nVar)
      {
        b(i, j, nVar) = U(i, offset + j, nVar);
      }
    }

  } // operator() - 2D

  template <DimensionType dimType_ = dimType>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dimType_ == THREE_D, int>::type & index) const
  {

    const int isize = U.extent(0);
    const int jsize = U.extent(1);
    const int ksize = U.extent(2);
    const int nbvar = U.extent(3);
    const int gw = ghostWidth;

    int i, j, k;

    // compute i,j,k
    if (boundaryLoc == XMIN or boundaryLoc == XMAX)
    {
      index2coord(index, i, j, k, gw, jsize, ksize);
    }
    else if (boundaryLoc == YMIN or boundaryLoc == YMAX)
    {
      index2coord(index, i, j, k, isize, gw, ksize);
    }
    else
    {
      index2coord(index, i, j, k, isize, jsize, gw);
    }

    /*
     * Proceed with copy
     */
    int offset = ghostWidth;
    if (boundaryLoc == XMAX)
      offset = U.extent(0) - 2 * ghostWidth;
    if (boundaryLoc == YMAX)
      offset = U.extent(1) - 2 * ghostWidth;
    if (boundaryLoc == ZMAX)
      offset = U.extent(2) - 2 * ghostWidth;


    /*
     * simple copy when PERIODIC or COPY
     */
    if (boundaryLoc == XMIN or boundaryLoc == XMAX)
    {

      for (int nVar = 0; nVar < nbvar; ++nVar)
      {
        b(i, j, k, nVar) = U(offset + i, j, k, nVar);
      }
    }
    else if (boundaryLoc == YMIN or boundaryLoc == YMAX)
    {

      for (int nVar = 0; nVar < nbvar; ++nVar)
      {
        b(i, j, k, nVar) = U(i, offset + j, k, nVar);
      }
    }
    else if (boundaryLoc == ZMIN or boundaryLoc == ZMAX)
    {

      for (int nVar = 0; nVar < nbvar; ++nVar)
      {
        b(i, j, k, nVar) = U(i, j, offset + k, nVar);
      }

    } // end (boundaryLoc == ZMIN or boundaryLoc == ZMAX)

  } // operator() - 3D

  DataArray b;
  DataArray U;
  int       ghostWidth;

}; // class CopyDataArray_To_BorderBuf

} // namespace euler_kokkos

#endif // MPI_BORDER_UTILS_H_
