set (config_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/INIReader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/INIReader.h
  )

add_library(config
  ${config_SRCS}
  )
target_include_directories(config
  PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/inih ) 
