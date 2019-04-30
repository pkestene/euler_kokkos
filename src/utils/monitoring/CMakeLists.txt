find_package(OpenMP)

add_library(monitoring "")

target_sources(monitoring
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/OpenMPTimer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OpenMPTimer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTimer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTimer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaTimer.h
  )

if(OpenMP_CXX_FOUND)
  target_link_libraries(monitoring PUBLIC OpenMP::OpenMP_CXX)
endif()
