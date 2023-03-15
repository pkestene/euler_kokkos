# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.18)

#
# default local cmake macro repository
#
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#
# Prevent from build in source tree
#
include(preventBuildInSource)

#
# Create project version (using git info ?)
# TODO
#

#
# Init build type: Release, Debug, ...
#
include(initBuildType)

# options
option (EULER_KOKKOS_BUILD_DOC  "Enable / disable documentation build (sphinx/html)" OFF)
option (USE_SPHINX_EXHALE "Enable / disable building API documentation (very long)" OFF)
option (USE_MPI "Activate / want MPI build" OFF)
option (USE_VTK "Activate / want VTK build" OFF)
option (USE_DOUBLE "build with double precision" ON)
option (USE_HDF5 "build HDF5 input/output support" OFF)
option (USE_PNETCDF "build PNETCDF input/output support (MPI required)" OFF)
option (USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)
option (USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)

# disable base languages
unset(PROJECT_LANGUAGES)

set(PROJECT_LANGUAGES ${PROJECT_LANGUAGES} C CXX)

project(euler_kokkos
  LANGUAGES ${PROJECT_LANGUAGES})

# Documentation type
if(EULER_KOKKOS_BUILD_DOC)
  set(EULER_KOKKOS_DOC_TYPE "Undefined" CACHE STRING
    "The documentation type to generate. Available values are html and doxygen")
  # Set the possible values for documentation type
  set_property(CACHE EULER_KOKKOS_DOC_TYPE PROPERTY STRINGS "html" "doxygen" "Undefined")
  add_subdirectory(doc)
  return()
endif()

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++14 is for Kokkos
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# Write a small header with build data, git version, etc...
#
include(write_version)

#####################################################################
# External packages: MPI, ...
#####################################################################

#####################################################################
#find_package(MPI REQUIRED)
find_package(MPI)
if (USE_MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})

    #set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})
    find_program(OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)

    # Full command line to probe if cuda support in MPI implementation is enabled
    # ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
    if (OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO}
        OUTPUT_VARIABLE _output)
      if ( (_output MATCHES "smcuda") OR (USE_MPI_CUDA_AWARE_ENFORCED) )
        message(STATUS "Found OpenMPI with CUDA support built.")
      else()
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
	add_compile_options(-DMPI_CUDA_AWARE_OFF)
      endif()
    endif()
  else()
    message(WARNING "Not compiling with MPI. Suppress this warning with -DUSE_MPI=OFF")
    set(USE_MPI OFF)
  endif()
endif()


#####################################################################
# VTK configuration tips, see
# /usr/lib/cmake/vtk-6.2/VTKConfig.cmake
# /usr/lib/cmake/vtk-6.2/UseVTK.cmake
#####################################################################
if (USE_VTK)
  # look for VTK only if requested; VTK macro might even be not present
  # on the target platform
  find_package(VTK COMPONENTS
    CommonCore
    CommonDataModel
    CommonExecutionModel
    CommonMath
    CommonMisc
    CommonSystem
    CommonTransforms
    IOCore
    IOGeometry
    IOImage
    IOLegacy
    IOXML
    IOXMLParser
    ParallelCore
    ParallelMPI
    IOParallelXML)

  if (VTK_FOUND)
    message("VTK version: ${VTK_VERSION}")
  else()
    message ("VTK NOT FOUND")
  endif()
endif(USE_VTK)

#####################################################################
# HDF5
#####################################################################
# prefer using parallel HDF5 when build with mpi
if (USE_MPI)
  set(HDF5_PREFER_PARALLEL TRUE)
endif(USE_MPI)

if (USE_HDF5)
  find_package(HDF5)
  if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    set(my_hdf5_libs hdf5 hdf5_cpp)
    add_compile_options(-DUSE_HDF5)
    if (HDF5_IS_PARALLEL)
      add_compile_options(-DUSE_HDF5_PARALLEL)
    endif()
  endif(HDF5_FOUND)
endif(USE_HDF5)

#####################################################################
# PNETCDF
#####################################################################
if (USE_MPI)
  if (USE_PNETCDF)
    find_package(PNETCDF)
    if (PNETCDF_FOUND)
      add_compile_options(-DUSE_PNETCDF)
      include_directories(${PNETCDF_INCLUDE_DIRS})
    endif(PNETCDF_FOUND)
  endif(USE_PNETCDF)
endif(USE_MPI)

#
# Kokkos : https://github.com/kokkos/kokkos
#
include(build_or_find_kokkos)

#
# common flags
#

if (USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
endif()

if (USE_MPI)
  add_compile_options(-DUSE_MPI)
endif()

if (USE_FPE_DEBUG)
  add_compile_options(-DUSE_FPE_DEBUG)
endif()

##
## Using flags -Wextra, it's to strong for Kokkos, too many warnings
## But -Wall is really a minimum
##
#add_definitions(-Wall -Wextra -Wconversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wcast-qual -Wunused -Woverloaded-virtual  -pedantic )
#add_definitions( -Wall -Wextra )
add_definitions( -Wall )

#
# sources
#
add_subdirectory(test)
add_subdirectory(src)

##################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
                         "${CMAKE_CXX_COMPILER_VERSION} "
                         "${CMAKE_CXX_COMPILER_WRAPPER}")
message("                 ${CMAKE_CXX_COMPILER}")

message("  MPI enabled                  = ${USE_MPI}")

message("  Kokkos version               = ${Kokkos_VERSION}")
message("  Kokkos_CXX_COMPILER          = ${Kokkos_CXX_COMPILER}")
message("  Kokkos_CXX_COMPILER_ID       = ${Kokkos_CXX_COMPILER_ID}")
message("  Kokkos_CXX_STANDARD          = ${Kokkos_CXX_STANDARD}")
message("  Kokkos_OPTIONS               = ${Kokkos_OPTIONS}")
message("  Kokkos_TPLS                  = ${Kokkos_TPLS}")
message("  Kokkos_DIRS                  = ${Kokkos_DIR}")

if(Kokkos_ENABLE_OPENMP)
  message("  Kokkos_ENABLE_OPENMP         = ${Kokkos_ENABLE_OPENMP}")
endif()

if(Kokkos_ENABLE_CUDA)
  message("  Kokkos_ENABLE_CUDA           = ${Kokkos_ENABLE_CUDA}")
  message("  Kokkos_ENABLE_CUDA_LAMBDA    = ${Kokkos_CUDA_LAMBDA_ENABLED}")
  message("  Kokkos_ENABLE_CUDA_CONSTEXPR = ${Kokkos_CUDA_CONSTEXPR_ENABLED}")
  message("  Kokkos_ENABLE_CUDA_UVM       = ${Kokkos_CUDA_UVM_ENABLED}")
  message("  Kokkos CUDA   flags          = ${KOKKOS_CUDA_OPTIONS}")
  #message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
  #message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
  #message("  CUDA Compile flags : ${CUDA_NVCC_FLAGS}")
endif(Kokkos_ENABLE_CUDA)

if(Kokkos_ENABLE_HIP)
  message("  Kokkos_ENABLE_HIP            = ${Kokkos_ENABLE_HIP}")
endif(Kokkos_ENABLE_HIP)

message("  Kokkos_ENABLE_HWLOC          = ${Kokkos_TPLS_HWLOC_ENABLED}")

message("  Kokkos architecture          = ${Kokkos_ARCH}")

if (HDF5_FOUND)
  message("  HDF5_VERSION                 = ${HDF5_VERSION}")
  message("  HDF5_DEFINITIONS             = ${HDF5_DEFINITIONS}")
  message("  HDF5_IS_PARALLEL             = ${HDF5_IS_PARALLEL}")
  message("  HDF5_INCLUDE_DIRS            = ${HDF5_INCLUDE_DIRS}")
  message("  HDF5_LIBRARIES               = ${HDF5_LIBRARIES}")
endif(HDF5_FOUND)

if (PNETCDF_FOUND)
  message("  PNETCDF_VERSION_STRING       = ${PNETCDF_VERSION_STRING}")
  message("  PNETCDF_INCLUDE_DIRS         = ${PNETCDF_INCLUDE_DIRS}")
  message("  PNETCDF_LIBRARIES            = ${PNETCDF_LIBRARIES}")
endif(PNETCDF_FOUND)
message("")
