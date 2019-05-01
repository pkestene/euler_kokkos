#
# custom executable macro
#
macro(MPI_TEST_ADD_EXEC test_name)
  set(EXE_NAME ${test_name})
  add_executable(${EXE_NAME}
    ${EXE_NAME}.cpp
    )
  target_link_libraries(${EXE_NAME}
    PRIVATE
    EulerKokkos::monitoring 
    EulerKokkos::mpiUtils)
endmacro(MPI_TEST_ADD_EXEC)


#
# add some executables
#
MPI_TEST_ADD_EXEC(testMpiHello)
MPI_TEST_ADD_EXEC(testMpiHelloGlobalSession)
MPI_TEST_ADD_EXEC(testMpiTopology_2D_C_API)
MPI_TEST_ADD_EXEC(testMpiTopology_3D_C_API)
MPI_TEST_ADD_EXEC(testMpiTopology_2D)

# if VTK is activated and found
# build this test
if(USE_VTK)

  if (VTK_FOUND)
    MPI_TEST_ADD_EXEC(testVtkXMLPImageDataWriter)
    if(${VTK_MAJOR_VERSION} EQUAL 6 OR ${VTK_MAJOR_VERSION} GREATER 6)
      set_target_properties(testVtkXMLPImageDataWriter PROPERTIES
	COMPILE_FLAGS "-DHAVE_VTK6")
    endif()
    
    target_include_directories(testVtkXMLPImageDataWriter PUBLIC ${VTK_INCLUDE_DIRS})
    
    target_link_libraries(testVtkXMLPImageDataWriter ${VTK_LIBRARIES})
  endif(VTK_FOUND)
  
endif(USE_VTK)
