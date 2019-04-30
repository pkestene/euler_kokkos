add_library(config "")

target_sources(config
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/INIReader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/INIReader.h
  )

target_include_directories(config
  PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/inih ) 

if (USE_MPI)
  target_compile_options    (config PUBLIC ${MPI_CXX_COMPILE_FLAGS})
  target_include_directories(config PUBLIC ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries     (config PUBLIC ${MPI_CXX_LIBRARIES})
endif(USE_MPI)
