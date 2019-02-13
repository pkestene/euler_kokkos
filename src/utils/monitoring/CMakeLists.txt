find_package(OpenMP)

set (monitoring_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/OpenMPTimer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OpenMPTimer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTimer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTimer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaTimer.h
  )

add_library(monitoring
  ${monitoring_SRCS}
  )

if(OpenMP_CXX_FOUND)
  target_link_libraries(monitoring PUBLIC OpenMP::OpenMP_CXX)
endif()
