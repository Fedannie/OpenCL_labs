cmake_minimum_required (VERSION 3.1)
project(prefix)

set(CMAKE_CXX_STANDARD 11)

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable (prefix main.cpp prefix.cl)
target_link_libraries (prefix ${OpenCL_LIBRARY})

# copy cl files to binary folder
FILE(GLOB CL_FILES *.cl)
foreach(cl_file ${CL_FILES})
    add_custom_command(
    TARGET prefix
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy ${cl_file} ${PROJECT_BINARY_DIR}/.
)
endforeach(cl_file)