# this macro is borrowed to https://github.com/melvinvermeeren/template_cpp
# under unlicence
function(find_python_module module module_u)

  # set(MODULE_U ${module_u})

  if(PY_${module_u}_FOUND)
    return()
  endif()

  # do not override user-specified values
  if(NOT PY_${module_u}_PATH)
    # set the default in case nothing is found
    set(PY_${module_u}_PATH
        ""
        CACHE STRING "Path to ${module}." FORCE)

    # A module's location is usually a directory, but for binary modules it's an .so file.
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" "-c" "import os, re, ${module}; print(os.path.dirname(\
               re.compile('/__init__.py.*').sub('',${module}.__file__)))"
      RESULT_VARIABLE "PY_${module_u}_RESULT"
      OUTPUT_VARIABLE "PY_${module_u}_OUTPUT"
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    # strip module directories to get the base path abc.foo.bar.xyz -> ... (three dots)
    string(REGEX REPLACE "[^\\.]+" "" module_strip "${module}")
    string(LENGTH "${module_strip}" module_strip)
    # strip ${module_strip} times the final directory from path
    if(module_strip GREATER 0)
      # CMake's loops are inclusive so start at 1
      foreach(module_strip_i RANGE 1 ${module_strip})
        string(REGEX REPLACE "/[^/]+$" "" "PY_${module_u}_OUTPUT" "${PY_${module_u}_OUTPUT}")
      endforeach(module_strip_i)
    endif()

    # if the exit code (RESULT) is non-zero python couldn't import module
    if(NOT PY_${module_u}_RESULT)
      set(PY_${module_u}_PATH
          "${PY_${module_u}_OUTPUT}"
          CACHE STRING "Path to ${module}." FORCE)
    endif()
  endif()

  if(NOT PY_${module_u}_PATH)
    message(FATAL_ERROR "Could not find ${module}.")
  endif()
  set(PY_${module_u}_FOUND
      ON
      CACHE BOOL "Found ${module}.")
  mark_as_advanced(PY_${module_u}_FOUND)
  message(STATUS "Found ${module}: ${PY_${module_u}_PATH}")

endfunction(find_python_module)
