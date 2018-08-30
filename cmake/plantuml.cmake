if(PLANTUML_FOUND)
	return()
endif()

find_program(PLANTUML_PATH
	NAMES plantuml plantuml.bat
	DOC "Path to PlantUML wrapper script.")

if(NOT PLANTUML_PATH)
	message(FATAL_ERROR "Could not find PlantUML.")
endif()
set(PLANTUML_FOUND ON CACHE BOOL "Found PlantUML.")
mark_as_advanced(PLANTUML_FOUND)
message(STATUS "Found PlantUML: ${PLANTUML_PATH}")
