cmake_minimum_required(VERSION 3.1)

project(
  contrast
  VERSION 1.0
  LANGUAGES CXX)


find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(contrast contrast-enhancement.cpp openmp-histogram.cpp  contrast.cpp)

target_link_libraries(contrast ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
