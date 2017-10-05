cmake_minimum_required(VERSION 3.1)

project(CombBLAS CXX)

set(CombBLAS_VERSION 1.5.0)

include(GenerateExportHeader)

# MPI
find_package(MPI REQUIRED)

# OpenMP
find_package(OpenMP)

# include_directories(third-party)
add_library(CombBLAS src/CommGrid.cpp src/mmio.cpp src/MPIType.cpp src/MPIOp.cpp src/MemoryPool.cpp src/hash.cpp)
target_compile_features(CombBLAS INTERFACE cxx_std_11)
target_link_libraries(CombBLAS MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
  add_definitions(-DTHREADED)
  target_link_libraries(CombBLAS OpenMP:OpenMP_CXX)
endif()
generate_export_header(CombBLAS)
set_property(TARGET CombBLAS PROPERTY VERSION ${CombBLAS_VERSION})
set_property(TARGET CombBLAS PROPERTY SOVERSION 3)
set_property(TARGET CombBLAS PROPERTY
  INTERFACE_CombBLAS_MAJOR_VERSION 3)
set_property(TARGET CombBLAS APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING CombBLAS_MAJOR_VERSION
)
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

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
