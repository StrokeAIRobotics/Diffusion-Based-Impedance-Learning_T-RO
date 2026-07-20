cmake_minimum_required(VERSION 3.16)

project(AVP_stream
  VERSION 0.1.0
  LANGUAGES CXX)

# ---- Language / features (keep 11 if you truly need it; 17 is a safer default nowadays)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Options to keep deps optional & portable
option(AVP_USE_PYTHON "Enable Python embedding (Python3::Python)" ON)
option(AVP_USE_QT     "Enable Qt concurrency utilities (Qt[6|5]::Core/Concurrent)" ON)

# ---- Install dirs (portable bin/lib/include paths)
include(GNUInstallDirs)

# ---- Sources (prefer listing explicitly; GLOB is fine for quick publication)
file(GLOB_RECURSE AVP_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME} ${AVP_SOURCES})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# ---- Dependencies (prefer imported targets)
# Boost (filesystem/system/thread/date_time/chrono)
find_package(Boost REQUIRED COMPONENTS filesystem system thread date_time chrono)
target_link_libraries(${PROJECT_NAME} PRIVATE
  Boost::filesystem Boost::system Boost::thread Boost::date_time Boost::chrono)

# Python (optional)
if(AVP_USE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
  target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
endif()

# Qt (optional; try Qt6, then fallback to Qt5)
if(AVP_USE_QT)
  find_package(Qt6 QUIET COMPONENTS Core Concurrent)
  if(Qt6_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core Qt6::Concurrent)
  else()
    find_package(Qt5 REQUIRED COMPONENTS Core Concurrent)
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Core Qt5::Concurrent)
  endif()
endif()

# ---- Explicit-FRI-style local libraries (no hard-coded paths)
# Users can pass roots: -DEXPLICIT_CPP_ROOT=/path -DFRI_ROOT=/path -DFTSENSOR_ROOT=/path
set(EXPLICIT_CPP_ROOT "" CACHE PATH "Root of Explicit-cpp install or build")
set(FRI_ROOT          "" CACHE PATH "Root of FRI client install or build")
set(FTSENSOR_ROOT     "" CACHE PATH "Root of FTSensor install or build")

# Helper macro to create imported targets when there's no find_package config
function(_add_local_lib libName libFileName rootVar)
  if(NOT "${${rootVar}}" STREQUAL "")
    # Try typical lib/include subfolders under the provided root
    find_library(${libName}_LIB
      NAMES ${libFileName}
      HINTS "${${rootVar}}"
      PATH_SUFFIXES lib lib64)
    find_path(${libName}_INC
      NAMES
        # put at least one header that exists; fallback to include dir
        .
      HINTS "${${rootVar}}"
      PATH_SUFFIXES include)
    if(${libName}_LIB)
      add_library(${libName} UNKNOWN IMPORTED)
      set_target_properties(${libName} PROPERTIES
        IMPORTED_LOCATION "${${libName}_LIB}")
      if(${libName}_INC)
        set_target_properties(${libName} PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${${libName}_INC}")
      endif()
      target_link_libraries(${PROJECT_NAME} PRIVATE ${libName})
    endif()
  endif()
endfunction()

# Example names; adjust if your actual filenames differ
# e.g., libExplicit-cpp.so, libFRIClient.a, libFTSensor.so
_add_local_lib(EXPLICIT_CPP  "Explicit-cpp" EXPLICIT_CPP_ROOT)
_add_local_lib(FRI_CLIENT    "FRIClient"    FRI_ROOT)
_add_local_lib(FTSENSOR      "FTSensor"     FTSENSOR_ROOT)

# ---- Warnings (portable defaults)
if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Install rules
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ---- Export a basic package config (optional but nice for downstreams)
include(CMakePackageConfigHelpers)
set(export_namespace AVP)
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets)
install(EXPORT ${PROJECT_NAME}Targets
        NAMESPACE ${export_namespace}::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  @ONLY)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})