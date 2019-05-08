find_package(OpenMP)

add_library(monitoring "")

target_sources(monitoring
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/OpenMPTimer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTimer.cpp
  )

target_include_directories(monitoring
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(monitoring PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)

#Add an alias so that library can be used inside the build tree
add_library(EulerKokkos::monitoring ALIAS monitoring)
