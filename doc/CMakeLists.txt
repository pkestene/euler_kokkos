find_package(Doxygen REQUIRED COMPONENTS dot)

get_target_property(DOXY_DOT_PATH Doxygen::dot IMPORTED_LOCATION)
get_filename_component(DOXY_DOT_PATH "${DOXY_DOT_PATH}" DIRECTORY)
file(TO_NATIVE_PATH "${DOXY_DOT_PATH}" DOXY_DOT_PATH)

set(DOC_SRC_DIR "${CMAKE_SOURCE_DIR}/doc")
set(DOC_BIN_DIR "${CMAKE_BINARY_DIR}/doc")
set(DOC_DEST_DIR "${CMAKE_BINARY_DIR}/doc/${DOC}")
#set(DOC_TMP "${CMAKE_BINARY_DIR}/tmp/doc")

set(DOXY_WARN_AS_ERR "NO")

set(SPHINX_LOG_FILE ${CMAKE_BINARY_DIR}/doc/sphinx_warnings.log)

# if the target is doxygen build only doxygen HTML
if(DOC STREQUAL "doxygen")

  set(DOXY_BUILD_HTML "YES")
  set(DOXY_BUILD_XML "NO")
  set(DOXY_OUTPUT_DIR "${DOC_DEST_DIR}")
  
# otherwise generate only XML and configure sphinx
else()

  if(DOC STREQUAL "html")
    set(EXTRA_PYTHON_MODULES "sphinx_rtd_theme")
  else()
    unset(EXTRA_PYTHON_MODULES)
  endif()
  
  set(DOXY_BUILD_HTML "NO")
  set(DOXY_BUILD_XML "YES")
  set(DOXY_OUTPUT_DIR "${DOC_DEST_DIR}/doxygen")
  
  # https://git.mel.vin/template/c/issues/50
  # https://bugzilla.gnome.org/show_bug.cgi?id=792676
  set(DOXY_WARN_AS_ERR "NO")

  if(USE_SPHINX_EXHALE)
    set(EXHALE_API_COMMAND "api/${PROJECT_NAME}_root")
  endif()
  
  # copy rst files into build dir
  add_custom_target(copy_files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/doc" "${CMAKE_CURRENT_BINARY_DIR}/doc" )
  
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/index.rst"
    "${CMAKE_CURRENT_BINARY_DIR}/index.rst")
  
  find_package(PythonInterp REQUIRED)
  include(find_python_module)
  
  # do not change the ENV, doesn't work with add_custom_target
  # https://cmake.org/Bug/view.php?id=5145
  if(ENV{PYTHONPATH})
    set(PYTHONPATH "$ENV{PYTHONPATH}")
  else()
    # PYTHONPATH disables defaults, manually append them
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
      "import sys; sys.stdout.write(':'.join(sys.path))"
      OUTPUT_VARIABLE PYTHONPATH)
  endif()
  
  foreach(module
      breathe
      sphinx
      sphinxcontrib.plantuml
      ${EXTRA_PYTHON_MODULES})
    string(TOUPPER "${module}" module_upper)
    find_python_module("${module}" "${module_upper}")
    set(PYTHONPATH "${${module_upper}_PATH}:${PYTHONPATH}")
  endforeach(module)
  
  if(${WERROR})
    set(SPHINX_WERROR "-W")
  else()
    unset(SPHINX_WERROR)
  endif()
  
  include(plantuml)

  if (USE_SPHINX_EXHALE)
    configure_file(
      "${DOC_SRC_DIR}/conf.py.in.exhale"
      "${DOC_BIN_DIR}/conf.py"
      @ONLY)
  else()
    configure_file(
      "${DOC_SRC_DIR}/conf.py.in"
      "${DOC_BIN_DIR}/conf.py"
      @ONLY)
  endif()
  
endif()

configure_file(
  "${DOC_SRC_DIR}/doxyfile.in"
  "${DOC_BIN_DIR}/doxyfile"
  @ONLY)

if(DOC STREQUAL "doxygen")
  add_custom_target(doc
    COMMAND Doxygen::doxygen "${DOC_BIN_DIR}/doxyfile"
    COMMENT "Generating HTML API documentation with Doxygen")
  
  add_custom_command(TARGET doc POST_BUILD
    COMMAND ;
    COMMENT "Output at ${DOC_DEST_DIR}/html/index.html")

  add_custom_target(clean_doc
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${DOC_DEST_DIR}")
  
else()

  # # List and copy all rst files from source dir to binary dir
  # file(GLOB_RECURSE files_rst RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  #   ${CMAKE_CURRENT_SOURCE_DIR}/*.rst)
  # foreach(filepath ${files_rst})
    
  #   configure_file (
  #     "${CMAKE_CURRENT_SOURCE_DIR}/${filepath}"
  #     "${CMAKE_CURRENT_BINARY_DIR}/${filepath}" COPYONLY)
  # endforeach()  
  
  configure_file(
    "${DOC_SRC_DIR}/doxyfile.in"
    "${DOC_DEST_DIR}/doxyfile"
    @ONLY)
  
  add_custom_target(doc

    DEPENDS copy_files
    
    COMMAND "${CMAKE_COMMAND}" -E echo
    "Generating Doxygen XML API documentation"

    COMMAND Doxygen::doxygen "${DOC_DEST_DIR}/doxyfile"
    
    COMMAND "${CMAKE_COMMAND}" -E echo
    "Generating sphinx ${DOC} documentation"
    # COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PYTHONPATH}"
    # "${PYTHON_EXECUTABLE}" -m sphinx -w ${SPHINX_LOG_FILE} -b "${DOC}"
    # -c "${DOC_DEST_DIR}" -q ${SPHINX_WERROR}
    # "${DOC_SRC_DIR}" "${DOC_DEST_DIR}")
  
    COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PYTHONPATH}"
    "${PYTHON_EXECUTABLE}" -m sphinx -w ${SPHINX_LOG_FILE} -b "${DOC}"
    -c "${DOC_BIN_DIR}" -q ${SPHINX_WERROR}
    "${DOC_BIN_DIR}" "${DOC_DEST_DIR}")
  
  add_custom_command(TARGET doc POST_BUILD
    COMMAND ;
    COMMENT "Output at ${DOC_DEST_DIR}")

  add_custom_target(clean_doc
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${DOC_DEST_DIR}")

endif()
