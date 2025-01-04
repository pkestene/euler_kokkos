# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# perf-dump Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ##############################################################################
# This file was found and adapted from original LLNL tools called perf-dump.
#
# For details, see https://github.com/LLNL/perf-dump
#
# ##############################################################################
#
# Try to find PAPI headers and libraries.
#
# Usage of this module as follows:
#
# find_package(PAPI)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
# * PAPI_ROOT : Set this environment variable to the root installation of
#   libpapi if the module has problems finding the proper installation path.
#
# Variables defined by this module:
#
# * PAPI_FOUND              System has PAPI libraries and headers
# * PAPI_LIBRARY            The PAPI library
# * PAPI_INCLUDE_DIR        The location of PAPI headers

find_library(
  PAPI_LIBRARY
  NAMES libpapi.so libpapi.a papi
  HINTS ENV PAPI_ROOT
  PATH_SUFFIXES lib lib64)

find_path(
  PAPI_INCLUDE_DIR
  NAMES papi.h
  HINTS ENV PAPI_ROOT
  PATH_SUFFIXES include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI REQUIRED_VARS PAPI_LIBRARY
                                                     PAPI_INCLUDE_DIR)

mark_as_advanced(PAPI_LIBRARY PAPI_INCLUDE_DIR)
