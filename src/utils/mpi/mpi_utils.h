#ifndef EULER_KOKKOS_UTILS_MPI_MPI_UTILS_H_
#define EULER_KOKKOS_UTILS_MPI_MPI_UTILS_H_

#include <mpi.h>

#include <cstdio>

namespace euler_kokkos
{

//! inline function checking the result of a MPI API call
int inline mpi_check_error(int                err,
                           char const * const func,
                           const char * const file,
                           int const          line)
{

  if (err != MPI_SUCCESS)
  {
    int  errorStringLen;
    char errorString[MPI_MAX_ERROR_STRING];
    MPI_Error_string(err, errorString, &errorStringLen);
    std::fprintf(stderr, "Error at %s:%d: calling %s ==> %s\n", file, line, func, errorString);
  }

  return err;
} // check_mpi_error

//! preprocessor macro ensuring that the error message can print filen name and line number
#define CHECK_MPI_ERR(value) mpi_check_error((value), #value, __FILE__, __LINE__)

} // namespace euler_kokkos

#endif // EULER_KOKKOS_UTILS_MPI_MPI_UTILS_H_
