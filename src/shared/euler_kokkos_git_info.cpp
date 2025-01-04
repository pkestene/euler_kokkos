#include <shared/euler_kokkos_git_info.h>

#include <euler_kokkos_version.h>

#include <iostream>

namespace euler_kokkos
{

std::string
GitRevisionInfo::version()
{
  return EULER_KOKKOS_VERSION;
}

bool
GitRevisionInfo::has_git_info()
{
#ifdef EULER_KOKKOS_HAS_GIT_INFO
  return true;
#else
  return false;
#endif
}

std::string
GitRevisionInfo::git_tag()
{
#ifdef EULER_KOKKOS_HAS_GIT_INFO
  return EULER_KOKKOS_GIT_TAG;
#else
  return "unknown tag";
#endif
}

std::string
GitRevisionInfo::git_head()
{
#ifdef EULER_KOKKOS_HAS_GIT_INFO
  return EULER_KOKKOS_GIT_HEAD;
#else
  return "unknown head";
#endif
}

std::string
GitRevisionInfo::git_hash()
{
#ifdef EULER_KOKKOS_HAS_GIT_INFO
  return EULER_KOKKOS_GIT_HASH;
#else
  return "unknown hash";
#endif
}

std::string
GitRevisionInfo::git_remote_url()
{
#ifdef EULER_KOKKOS_HAS_GIT_INFO
  return EULER_KOKKOS_GIT_REMOTE_URL;
#else
  return "unknown remote url";
#endif
}

std::string
GitRevisionInfo::git_branch()
{
#ifdef EULER_KOKKOS_HAS_GIT_INFO
  return EULER_KOKKOS_GIT_BRANCH;
#else
  return "unknown git branch";
#endif
}

bool
GitRevisionInfo::git_is_clean()
{
#ifdef EULER_KOKKOS_HAS_GIT_INFO
  return EULER_KOKKOS_GIT_IS_CLEAN;
#else
  return false;
#endif
}

void
GitRevisionInfo::print()
{
  if (has_git_info())
  {
    std::cout << "#############################################\n";
    std::cout << "euler_kokkos - git information" << "\n";
    std::cout << "git remote url : " << git_remote_url() << "\n";
    std::cout << "git branch     : " << git_branch() << "\n";
    std::cout << "git head       : " << git_head() << "\n";
    std::cout << "git hash       : " << git_hash() << " (";
    if (git_is_clean())
    {
      std::cout << "clean";
    }
    else
    {
      std::cout << "dirty";
    }
    std::cout << ")\n";
    std::cout << "#############################################\n";
  }
  else
  {
    std::cout << "#############################################\n";
    std::cout << "euler_kokkos - not built from a git repository      \n";
    std::cout << "version       : " << version() << "\n";
    std::cout << "#############################################\n";
  }

} // GitRevisionInfo::print

} // namespace euler_kokkos
