find_package(Git QUIET)
if (NOT GIT_FOUND)
  set(GIT_BUILD_STRING "N/A")
else()
  # check if we are building source from a git clone or a release tarball
  
  # try to run quietly a git command
  execute_process(COMMAND git log --pretty=format:'%h' -n 1
    OUTPUT_VARIABLE GIT_REV
    ERROR_QUIET)

  if ("${GIT_REV}" STREQUAL "")
    # set default values
    set(GIT_BUILD_STRING "N/A")
    set(GIT_BRANCH       "N/A")
    set(GIT_REMOTE       "N/A")
    set(GIT_REMOTE_URL   "N/A")
  else()
    # use git to populate target variables: GIT_BUILD_STRING, etc...
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
      OUTPUT_VARIABLE GIT_BUILD_STRING
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_BRANCH
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GIT_EXECUTABLE} config --get branch.${GIT_BRANCH}.remote
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_REMOTE
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GIT_EXECUTABLE} config --get remote.${GIT_REMOTE}.url
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_REMOTE_URL
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
endif()
execute_process(COMMAND date "+%d/%m/%y"
  OUTPUT_VARIABLE DATE_STRING
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND date "+%H:%M:%S"
  OUTPUT_VARIABLE TIME_STRING
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(RELEASE_BUILD True)
endif()

configure_file(
  ${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}_version.h.in
  ${PROJECT_BINARY_DIR}/src/${PROJECT_NAME}_version.h)
