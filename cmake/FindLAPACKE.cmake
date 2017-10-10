# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindLAPACKE
# -------------
#
# Find LAPACKE
#
# Find the LAPACKE C library
#
# Using LAPACKE:
#
# ::
#
#   find_package(LAPACKE REQUIRED)
#   include_directories(${LAPACKE_INCLUDE_DIRS})
#   add_executable(foo foo.cc)
#   target_link_libraries(foo ${LAPACKE_LIBRARIES})
#
# This module sets the following variables:
#
# ::
#
#   LAPACKE_FOUND - set to true if the library is found
#   LAPACKE_INCLUDE_DIRS - list of required include directories
#   LAPACKE_LIBRARIES - list of libraries to be linked

# UNIX paths are standard, no need to write.
find_library(LAPACKE_LIBRARY
  NAMES lapacke
  PATHS "$ENV{LAPACKE_LIB_DIR}"
  )
find_path(LAPACKE_INCLUDE_DIR
  NAMES lapacke.h
  PATHS "$ENV{LAPACKE_INC_DIR}"
  )

# ------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Lapacke
  REQUIRED_VARS LAPACKE_LIBRARY LAPACKE_INCLUDE_DIR)

if (LAPACKE_FOUND)
  set(LAPACKE_INCLUDE_DIRS ${LAPACKE_INCLUDE_DIR})
  set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY})
endif ()

# Hide internal variables
mark_as_advanced(
  LAPACKE_INCLUDE_DIR
  LAPACKE_LIBRARY)

