#include <utils/mpi/MpiComm.h>
#include <cstdint>
#include <cstdlib> // for std::abort
#include <array>
#include <vector>

namespace euler_kokkos
{

// ============================================================
// ============================================================
MpiComm::MpiComm()
{
  comm_ptr.reset(new MPI_Comm(MPI_COMM_WORLD));
}

// ============================================================
// ============================================================
MpiComm::MpiComm(const MPI_Comm & comm, comm_create_kind kind)
{
  if (comm == MPI_COMM_NULL)
    /* MPI_COMM_NULL indicates that the communicator is not usable. */
    return;

  switch (kind)
  {
    case COMM_DUPLICATE: {
      MPI_Comm newcomm;
      CHECK_MPI_ERR(MPI_Comm_dup(comm, &newcomm));
      comm_ptr.reset(new MPI_Comm(newcomm), comm_free());
      MPI_Comm_set_errhandler(newcomm, MPI_ERRORS_RETURN);
      break;
    }

    case COMM_TAKE_OWNERSHIP:
      comm_ptr.reset(new MPI_Comm(comm), comm_free());
      break;

    case COMM_ATTACH:
      comm_ptr.reset(new MPI_Comm(comm));
      break;
  }

} // MpiComm::MpiComm

// ============================================================
// ============================================================
MpiComm::operator MPI_Comm() const
{
  if (comm_ptr)
    return *comm_ptr;
  else
    return MPI_COMM_NULL;
} // MpiComm::operator MPI_Comm

// ============================================================
// ============================================================
void
MpiComm::MPI_Barrier() const
{
  CHECK_MPI_ERR(::MPI_Barrier(MPI_Comm(*this)));
}

// ============================================================
// ============================================================
void
MpiComm::abort(int errcode) const
{
  CHECK_MPI_ERR(::MPI_Abort(MPI_Comm(*this), errcode));
  std::abort();
}

} // namespace euler_kokkos
