cmake_minimum_required(VERSION 3.1)

PROJECT(CombBLAS C CXX)

# require c++11
set(CMAKE_CXX_STANDARD 11)

# MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
else (MPI_FOUND)
  message(SEND_ERROR "MPI is required")
endif (MPI_FOUND)

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else(OPENMP_FOUND)
    message(SEND_ERROR "OpenMP is required")
endif()

include_directories(include)
include_directories(third-party)
add_library(CombBLAS src/Commgrid.cpp src/mmio.c src/MPIType.cpp src/MPIOp.cpp src/MemoryPool.cpp src/hash.cpp)
target_link_libraries(CombBLAS ${MPI_CXX_LIBRARIES})

# installation
install(DIRECTORY include/ DESTINATION include)
install(TARGETS CombBLAS ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
