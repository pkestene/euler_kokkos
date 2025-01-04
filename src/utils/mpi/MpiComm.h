/**
 * \file MpiComm.h
 * \brief Object representation of a MPI communicator
 *
 * the following class is loosely adapted from boost::mpi package
 * see class communicator
 * https://www.boost.org/doc/libs/1_80_0/doc/html/mpi.html
 *
 * # Distributed under the Boost Software License, Version 1.0.
 * # https://www.boost.org/LICENSE_1_0.txt
 *
 */
#ifndef EULER_KOKKOS_UTILS_MPI_MPICOMM_H
#define EULER_KOKKOS_UTILS_MPI_MPICOMM_H

#include <utils/mpi/mpi_utils.h>

#include <mpi.h>
#include <memory> // for std::shared_ptr
#include <vector>

#include <cassert>
#ifndef assertm
#  define assertm(exp, msg) assert(((void)msg, exp))
#endif

namespace euler_kokkos
{

namespace MpiComm_impl
{

template <typename T>
inline MPI_Datatype
mpi_type()
{
  static_assert(!std::is_same<T, T>::value, "Unknown MPI type");
  return MPI_DATATYPE_NULL;
}

template <>
inline MPI_Datatype
mpi_type<double>()
{
  return MPI_DOUBLE;
}

template <>
inline MPI_Datatype
mpi_type<float>()
{
  return MPI_FLOAT;
}

template <>
inline MPI_Datatype
mpi_type<uint64_t>()
{
  return MPI_UINT64_T;
}

template <>
inline MPI_Datatype
mpi_type<uint32_t>()
{
  return MPI_UINT32_T;
}

template <>
inline MPI_Datatype
mpi_type<uint16_t>()
{
  return MPI_UINT16_T;
}

template <>
inline MPI_Datatype
mpi_type<uint8_t>()
{
  return MPI_UINT8_T;
}

template <>
inline MPI_Datatype
mpi_type<int64_t>()
{
  return MPI_INT64_T;
}

template <>
inline MPI_Datatype
mpi_type<int32_t>()
{
  return MPI_INT32_T;
}

template <>
inline MPI_Datatype
mpi_type<int16_t>()
{
  return MPI_INT16_T;
}

template <>
inline MPI_Datatype
mpi_type<int8_t>()
{
  return MPI_INT8_T;
}

template <>
inline MPI_Datatype
mpi_type<char>()
{
  return MPI_CHAR;
}

template <>
inline MPI_Datatype
mpi_type<bool>()
{
  return MPI_CXX_BOOL;
}

} // namespace MpiComm_impl

/**
 * \brief Object representation of an MPI communicator.
 *
 * Inspired by boost::mpi and trilinos teuchos packages.
 */
class MpiComm
{

public:
  //! enumerate possible operation used in MPI reduction / scan
  enum struct MPI_OP : int
  {
    MIN,
    MAX,
    SUM,
    PROD,
    LOR,
    BOR,
    LAND,
    BAND,
    NUM_OPS // nor a valid operation, just the total number of operations
  };

  static constexpr auto MIN = MPI_OP::MIN;
  static constexpr auto MAX = MPI_OP::MAX;
  static constexpr auto SUM = MPI_OP::SUM;
  static constexpr auto PROD = MPI_OP::PROD;
  static constexpr auto LOR = MPI_OP::LOR;
  static constexpr auto BOR = MPI_OP::BOR;
  static constexpr auto LAND = MPI_OP::LAND;
  static constexpr auto BAND = MPI_OP::BAND;

  //! mapping an MPI_Op_enum to an MPI_Op
  template <MPI_OP mpi_op>
  static MPI_Op
  MapMpiOp()
  {
    if constexpr (mpi_op == MPI_OP::MIN)
      return MPI_MIN;
    else if constexpr (mpi_op == MPI_OP::MAX)
      return MPI_MAX;
    else if constexpr (mpi_op == MPI_OP::SUM)
      return MPI_SUM;
    else if constexpr (mpi_op == MPI_OP::PROD)
      return MPI_PROD;
    else if constexpr (mpi_op == MPI_OP::LOR)
      return MPI_LOR;
    else if constexpr (mpi_op == MPI_OP::BOR)
      return MPI_BOR;
    else if constexpr (mpi_op == MPI_OP::LAND)
      return MPI_LAND;
    else if constexpr (mpi_op == MPI_OP::BAND)
      return MPI_BAND;
    // no default value, we xant the compilation to fail if operation is not found
  }


  /**
   * @brief Enumeration used to describe how to adopt a C @c MPI_Comm into
   * a MpiComm communicator.
   *
   * \note this enum is adapted from class communicator found in Boost::mpi
   *
   * The values for this enumeration determine how a MpiComm
   * communicator will behave when constructed with an MPI
   * communicator. The options are:
   *
   *   - @c COMM_DUPLICATE: Duplicate the MPI_Comm communicator to
   *   create a new communicator (e.g., with MPI_Comm_dup). This new
   *   MPI_Comm communicator will be automatically freed when the
   *   MpiComm communicator (and all copies of it) is destroyed.
   *
   *   - @c COMM_TAKE_OWNERSHIP: Take ownership of the communicator. It
   *   will be freed automatically when all of the MpiComm
   *   communicators go out of scope. This option must not be used with
   *   MPI_COMM_WORLD.
   *
   *   - @c COMM_ATTACH: The MpiComm communicator will reference the
   *   existing MPI communicator but will not free it when the MpiComm
   *   communicator goes out of scope. This option should only be used
   *   when the communicator is managed by the final user or MPI library
   *   (e.g., MPI_COMM_WORLD).
   */
  enum comm_create_kind
  {
    COMM_DUPLICATE,
    COMM_TAKE_OWNERSHIP,
    COMM_ATTACH
  };

  /*
   * Build a new MpiComm communicator for @c MPI_COMM_WORLD.
   *
   * Constructs a MpiComm communicator that attaches to @c
   * MPI_COMM_WORLD. This is the equivalent of constructing with
   * @c (MPI_COMM_WORLD, comm_attach).
   */
  MpiComm();

  /**
   * Build a new MpiComm communicator based on the MPI communicator
   * @p comm.
   *
   * @p comm may be any valid MPI communicator. If @p comm is
   * MPI_COMM_NULL, an empty communicator (that cannot be used for
   * communication) is created and the @p kind parameter is
   * ignored. Otherwise, the @p kind parameters determines how the
   * MpiComm communicator will be related to @p comm:
   *
   *   - If @p kind is @c comm_duplicate, duplicate @c comm to create
   *   a new communicator. This new communicator will be freed when
   *   the MpiComm communicator (and all copies of it) is destroyed.
   *   This option is only permitted if @p comm is a valid MPI
   *   intracommunicator or if the underlying MPI implementation
   *   supports MPI 2.0 (which supports duplication of
   *   intercommunicators).
   *
   *   - If @p kind is @c comm_take_ownership, take ownership of @c
   *   comm. It will be freed automatically when all of the MpiComm
   *   communicators go out of scope. This option must not be used
   *   when @c comm is MPI_COMM_WORLD.
   *
   *   - If @p kind is @c comm_attach, this MpiComm communicator
   *   will reference the existing MPI communicator @p comm but will
   *   not free @p comm when the MpiComm communicator goes out of
   *   scope. This option should only be used when the communicator is
   *   managed by the user or MPI library (e.g., MPI_COMM_WORLD).
   */
  MpiComm(const MPI_Comm & comm, comm_create_kind kind);

  //! conversion operator to MPI_Comm
  operator MPI_Comm() const;

  //! return raw MPI_Comm identifier
  inline MPI_Comm
  get_MPI_Comm() const
  {
    return MPI_Comm(*this);
  }

  //! conversion operator to boolean
  operator bool() const { return (bool)comm_ptr; }

  /**
   * @brief Determine the rank of the executing process in a
   * communicator.
   *
   * This routine is equivalent to @c MPI_Comm_rank.
   *
   *   @returns The rank of the process in the communicator, which
   *   will be a value in [0, size())
   */
  inline int
  rank() const
  {
    int rank_;
    CHECK_MPI_ERR(::MPI_Comm_rank(MPI_Comm(*this), &rank_));
    return rank_;
  }

  /**
   * @brief Determine the number of processes in a communicator.
   *
   * This routine is equivalent to @c MPI_Comm_size.
   *
   *   @returns The number of processes in the communicator.
   */
  inline int
  size() const
  {
    int size_;
    CHECK_MPI_ERR(::MPI_Comm_size(MPI_Comm(*this), &size_));
    return size_;
  }

#ifdef barrier
  // Linux defines a function-like macro named "barrier". So, we need
  // to avoid expanding the macro when we define our barrier()
  // function. However, some C++ parsers (Doxygen, for instance) can't
  // handle this syntax, so we only use it when necessary.
  void(barrier)() const;
#else
  /**
   * @brief Wait for all processes within a communicator to reach the
   * barrier.
   *
   * This routine is a collective operation that blocks each process
   * until all processes have entered it, then releases all of the
   * processes "simultaneously". It is equivalent to @c MPI_Barrier.
   */
  void
  MPI_Barrier() const;
#endif

  /** Abort all tasks in the group of this communicator.
   *
   *  Makes a "best attempt" to abort all of the tasks in the group of
   *  this communicator. Depending on the underlying MPI
   *  implementation, this may either abort the entire program (and
   *  possibly return @p errcode to the environment) or only abort
   *  some processes, allowing the others to continue. Consult the
   *  documentation for your MPI implementation. This is equivalent to
   *  a call to @c MPI_Abort
   *
   *  @param errcode The error code to return from aborted processes.
   *  @returns Will not return.
   */
  void
  abort(int errcode) const;

protected:
  /**
   * INTERNAL ONLY
   *
   * Function object that frees an MPI communicator and deletes the
   * memory associated with it. Intended to be used as a deleter with
   * shared_ptr.
   */
  struct comm_free
  {
    void
    operator()(MPI_Comm * comm) const
    {
      assertm(comm != 0, "MPI communicator pointer can't be null");
      assertm(*comm != MPI_COMM_NULL, "MPI comminucator can't be MPI_COMM_NULL");
      int finalized;
      CHECK_MPI_ERR(MPI_Finalized(&finalized));
      if (!finalized)
        CHECK_MPI_ERR(MPI_Comm_free(comm));
      delete comm;
    }
  };

public:
  template <MPI_OP op, typename T>
  void
  MPI_Reduce(const T * sendbuf, T * recvbuf, int count, int root) const
  {
    using namespace MpiComm_impl;
    CHECK_MPI_ERR(
      ::MPI_Reduce(sendbuf, recvbuf, count, mpi_type<T>(), MapMpiOp<op>(), root, MPI_Comm(*this)));
  }

  template <MPI_OP op, typename T>
  void
  MPI_Allreduce(const T * sendbuf, T * recvbuf, int count) const
  {
    using namespace MpiComm_impl;
    CHECK_MPI_ERR(
      ::MPI_Allreduce(sendbuf, recvbuf, count, mpi_type<T>(), MapMpiOp<op>(), MPI_Comm(*this)));
  }

  template <MPI_OP op, typename T>
  void
  MPI_Scan(const T * sendbuf, T * recvbuf, int count) const
  {
    using namespace MpiComm_impl;
    CHECK_MPI_ERR(
      ::MPI_Scan(sendbuf, recvbuf, count, mpi_type<T>(), MapMpiOp<op>(), MPI_Comm(*this)));
  }

  template <MPI_OP op, typename T>
  void
  MPI_Exscan(const T * sendbuf, T * recvbuf, int count) const
  {
    using namespace MpiComm_impl;
    CHECK_MPI_ERR(
      ::MPI_Exscan(sendbuf, recvbuf, count, mpi_type<T>(), MapMpiOp<op>(), MPI_Comm(*this)));
  }

  template <typename T>
  void
  MPI_Allgather(const T * sendbuf, T * recvbuf, int count) const
  {
    using namespace MpiComm_impl;
    CHECK_MPI_ERR(::MPI_Allgather(
      sendbuf, count, mpi_type<T>(), recvbuf, count, mpi_type<T>(), MPI_Comm(*this)));
  }

  template <typename T>
  void
  MPI_Allgatherv_inplace(T * sendrecvbuf, int count) const
  {
    int comm_size = this->size();

    std::vector<int> counts(comm_size);
    this->MPI_Allgather(&count, counts.data(), 1);
    std::vector<int> displs(comm_size);
    for (int i = 1; i < comm_size; i++)
      displs[i] = displs[i - 1] + counts[i - 1];

    using namespace MpiComm_impl;
    CHECK_MPI_ERR(::MPI_Allgatherv(MPI_IN_PLACE,
                                   0,
                                   0,
                                   sendrecvbuf,
                                   counts.data(),
                                   displs.data(),
                                   mpi_type<T>(),
                                   MPI_Comm(*this)));
  }

  template <typename T>
  void
  MPI_Bcast(T * buffer, int count, int root) const
  {
    using namespace MpiComm_impl;
    CHECK_MPI_ERR(::MPI_Bcast(buffer, count, mpi_type<T>(), root, MPI_Comm(*this)));
  }

  template <typename T>
  void
  MPI_Alltoall(const T * sendbuf, int sendcount, T * recvbuf, int recvcount) const
  {
    using namespace MpiComm_impl;
    CHECK_MPI_ERR(::MPI_Alltoall(
      sendbuf, sendcount, mpi_type<T>(), recvbuf, recvcount, mpi_type<T>(), MPI_Comm(*this)));
  }

  // NOTE : does not support sending to self
  template <typename Kokkos_View_t>
  MPI_Request
  MPI_Isend(const Kokkos_View_t & view, int dest, int tag) const
  {
    using namespace MpiComm_impl;
    MPI_Datatype type = mpi_type<typename Kokkos_View_t::value_type>();
    MPI_Request  r = MPI_REQUEST_NULL;
    CHECK_MPI_ERR(::MPI_Isend(view.data(), view.size(), type, dest, tag, MPI_Comm(*this), &r));
    return r;
  }

  template <typename Kokkos_View_t>
  MPI_Request
  MPI_Irecv(const Kokkos_View_t & view, int dest, int tag) const
  {
    using namespace MpiComm_impl;
    MPI_Datatype type = mpi_type<typename Kokkos_View_t::value_type>();
    MPI_Request  r = MPI_REQUEST_NULL;
    CHECK_MPI_ERR(::MPI_Irecv(view.data(), view.size(), type, dest, tag, MPI_Comm(*this), &r));
    return r;
  }

  inline void
  MPI_Waitall(int count, MPI_Request * requests) const
  {
    CHECK_MPI_ERR(::MPI_Waitall(count, requests, MPI_STATUSES_IGNORE));
  }

private:
  std::shared_ptr<MPI_Comm> comm_ptr;

}; // class MpiComm

} // namespace euler_kokkos

#endif // EULER_KOKKOS_UTILS_MPI_MPICOMM_H
