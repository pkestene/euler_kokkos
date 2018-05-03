add_executable(test_segfault
  test_segfault.cpp
  ${BACKWARD_ENABLE})
add_backward(test_segfault)
