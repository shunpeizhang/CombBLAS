cmake_minimum_required(VERSION 3.1)

project(CombBLAS CXX)

set(CombBLAS_VERSION 1.5.0)

include(GenerateExportHeader)

# require c++11
set(CMAKE_CXX_STANDARD 11)

set(is_cray "FALSE")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Cray")
    set(is_cray "TRUE")
endif()

# MPI
if(NOT is_cray)
  find_package(MPI REQUIRED)
endif()

# OpenMP
find_package(OpenMP REQUIRED)

# include_directories(third-party)
add_library(CombBLAS src/CommGrid.cpp src/mmio.cpp src/MPIType.cpp src/MPIOp.cpp src/MemoryPool.cpp src/hash.cpp)
generate_export_header(CombBLAS)
set_property(TARGET CombBLAS PROPERTY VERSION ${CombBLAS_VERSION})
set_property(TARGET CombBLAS PROPERTY SOVERSION 3)
set_property(TARGET CombBLAS PROPERTY
  INTERFACE_CombBLAS_MAJOR_VERSION 3)
set_property(TARGET CombBLAS APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING CombBLAS_MAJOR_VERSION
)
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
if(NOT is_cray)
  target_link_libraries(CombBLAS ${MPI_CXX_LIBRARIES})
  target_include_directories(CombBLAS PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()
target_compile_options(CombBLAS PUBLIC "${OpenMP_CXX_FLAGS}")

# installation
install(DIRECTORY include/ DESTINATION include)
install(TARGETS CombBLAS EXPORT CombBLASTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfigVersion.cmake"
  VERSION ${CombBLAS_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT CombBLASTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASTargets.cmake"
  NAMESPACE combblas::
)
configure_file(cmake/CombBLASConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfig.cmake"
  COPYONLY
)

set(ConfigPackageLocation lib/cmake/CombBLAS)
install(EXPORT CombBLASTargets
  FILE
    CombBLASTargets.cmake
  NAMESPACE
    comblas::
  DESTINATION
    ${ConfigPackageLocation}
)
install(
  FILES
    cmake/CombBLASConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfigVersion.cmake"
  DESTINATION
    ${ConfigPackageLocation}
  COMPONENT
    Devel
)
