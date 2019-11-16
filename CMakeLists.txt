# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.10)

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
option (BUILD_CODE "Enable / disable project build" ON)
option (BUILD_DOC  "Enable / disable documentation build (sphinx/html)" OFF)
option (USE_SPHINX_EXHALE "Enable / disable building API documentation (very long)" OFF)
option (USE_MPI "Activate / want MPI build" OFF)
option (USE_VTK "Activate / want VTK build" OFF)
option (USE_DOUBLE "build with double precision" ON)
option (USE_HDF5 "build HDF5 input/output support" OFF)
option (USE_PNETCDF "build PNETCDF input/output support (MPI required)" OFF)
option (USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)
option (USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)

# Documentation type
if(NOT DEFINED DOC)
  set(DOC OFF CACHE STRING
    "The documentation type to generate. Available values are html and doxygen" FORCE)
endif()

# disable base languages
unset(PROJECT_LANGUAGES)

if(${BUILD_CODE})
  set(PROJECT_LANGUAGES ${PROJECT_LANGUAGES} C CXX)
endif()

project(euler_kokkos
  LANGUAGES ${PROJECT_LANGUAGES})

# this allows to disable build code,
# as one might want to build only documentation
if(${BUILD_CODE})

  # always export compile commands database
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  
  # C++11 is for Kokkos
  set(CMAKE_CXX_STANDARD 11)
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
    find_package(VTK)
    
    # the following add VTK to all targets
    # if(VTK_FOUND)
    #   include(${VTK_USE_FILE})
    # endif(VTK_FOUND)
    if (VTK_FOUND)
      message("***VTK FOUND ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
    else()
      message ("*** VTK NOT FOUND")
    endIF()
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
      add_compile_options(-DUSE_HDF5 -DH5_NO_DEPRECATED_SYMBOLS)
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
  # backtrace does not build anymore with cuda 10.1
  #add_subdirectory(external/backward-cpp)
  add_subdirectory(external/kokkos)
  
  # pass Kokkos include directories to our target application
  include_directories(${Kokkos_INCLUDE_DIRS_RET})
  
  add_subdirectory(test)
  add_subdirectory(src)
endif()

if(NOT DOC STREQUAL "OFF")
  add_subdirectory(doc)
endif()
  
##################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} " 
                         "${CMAKE_CXX_COMPILER_VERSION} " 
                         "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
if (USE_MPI)
  message("  MPI enabled")
else(USE_MPI)
  message("  MPI not enabled")
endif(USE_MPI)
if (USE_CUDA)
  message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
  message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
  message("  CUDA Compile flags : ${CUDA_NVCC_FLAGS}")
else(USE_CUDA)
  message("  CUDA not enabled")
endif(USE_CUDA)
if (HDF5_FOUND)
  message("  HDF5 found version    : ${HDF5_VERSION}")
  message("  HDF5 definitions      : ${HDF5_DEFINITIONS}")
  message("  HDF5 parallel         : ${HDF5_IS_PARALLEL}")
  message("  HDF5 includes dirs    : ${HDF5_INCLUDE_DIRS}")
  message("  HDF5 libraries        : ${HDF5_LIBRARIES}")
endif(HDF5_FOUND)
if (PNETCDF_FOUND)
  message("  PNETCDF found version : ${PNETCDF_VERSION_STRING}")
  message("  PNETCDF include dirs  : ${PNETCDF_INCLUDE_DIRS}")
  message("  PNETCDF libraries     : ${PNETCDF_LIBRARIES}")
endif(PNETCDF_FOUND)
message("")
