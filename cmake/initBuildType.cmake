#
# Set default compile optimization flag
#
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
  message(
    STATUS
      "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${DEFAULT_BUILD_TYPE}"
      CACHE
        STRING
        "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
        FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()
