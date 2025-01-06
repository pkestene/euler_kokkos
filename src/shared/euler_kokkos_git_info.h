/**
 * \file euler_kokkos_git_info.h
 */
#ifndef EULER_KOKKOS_SHARED_EULER_KOKKOS_GIT_INFO_H_
#define EULER_KOKKOS_SHARED_EULER_KOKKOS_GIT_INFO_H_

#include <string>

namespace euler_kokkos
{

struct GitRevisionInfo
{
  //! cmake project version
  static std::string
  version();

  static bool
  has_git_info();

  static std::string
  git_tag();

  static std::string
  git_head();

  static std::string
  git_hash();

  static std::string
  git_remote_url();

  static std::string
  git_branch();

  static bool
  git_is_clean();

  static void
  print();
};

} // namespace euler_kokkos

#endif // EULER_KOKKOS_SHARED_EULER_KOKKOS_GIT_INFO_H_
