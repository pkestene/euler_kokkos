#
# custom executable macro
#
macro(MPI_TEST_ADD_EXEC test_name)
  set(EXE_NAME ${test_name})
  add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
  target_link_libraries(${EXE_NAME} PRIVATE Kokkos::kokkos EulerKokkos::monitoring
                                            EulerKokkos::mpiUtils)
endmacro(MPI_TEST_ADD_EXEC)

#
# add some executables
#
mpi_test_add_exec(testMpiHello)
mpi_test_add_exec(testMpiHelloGlobalSession)
mpi_test_add_exec(testMpiTopology_2D_C_API)
mpi_test_add_exec(testMpiTopology_3D_C_API)
mpi_test_add_exec(testMpiTopology_2D)

# if VTK is activated and found build this test
if(EULER_KOKKOS_USE_VTK)

  if(VTK_FOUND)
    mpi_test_add_exec(testVtkXMLPImageDataWriter)
    # cmake-format: off
    if(${VTK_MAJOR_VERSION} EQUAL 6 OR ${VTK_MAJOR_VERSION} GREATER 6)
      set_target_properties(
        testVtkXMLPImageDataWriter
        PROPERTIES COMPILE_FLAGS "-DHAVE_VTK6")
    endif()
    # cmake-format: on

    target_include_directories(testVtkXMLPImageDataWriter PUBLIC ${VTK_INCLUDE_DIRS})

    target_link_libraries(testVtkXMLPImageDataWriter PRIVATE ${VTK_LIBRARIES})
  endif(VTK_FOUND)

endif(EULER_KOKKOS_USE_VTK)
