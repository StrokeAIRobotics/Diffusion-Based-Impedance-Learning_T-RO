cmake_minimum_required(VERSION 3.16)

project(Parkour
  VERSION 0.1.0
  LANGUAGES CXX)

# ---- Language
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Options (toggle features without editing this file)
option(UP_USE_PYTHON "Enable Python embedding (Python3::Python)" OFF)
option(UP_USE_QT     "Enable Qt concurrency utilities (Qt[6|5])" OFF)
option(UP_USE_SERIAL "Link libSerial (for hand driver deps)"    OFF)
option(UP_USE_HAND   "Link qbrobotics hand driver library"      OFF)

# ---- Install dirs
include(GNUInstallDirs)

# ---- Sources (assumes src/ and include/ layout)
file(GLOB_RECURSE UP_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME} ${UP_SOURCES})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# ---- Boost (required)
find_package(Boost REQUIRED COMPONENTS filesystem system thread date_time chrono)
target_link_libraries(${PROJECT_NAME} PRIVATE
  Boost::filesystem Boost::system Boost::thread Boost::date_time Boost::chrono)

# ---- Python (optional)
if(UP_USE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()

# ---- Qt (optional; prefer Qt6 and fall back to Qt5)
if(UP_USE_QT)
  find_package(Qt6 QUIET COMPONENTS Core Concurrent)
  if(Qt6_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core Qt6::Concurrent)
  else()
    find_package(Qt5 REQUIRED COMPONENTS Core Concurrent)
    target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Core Qt5::Concurrent)
  endif()
endif()

# ---- External robotics libs (provide roots or let CMAKE_PREFIX_PATH find them)
# Users can pass:
#   -DEXPLICIT_CPP_ROOT=/path
#   -DFRI_ROOT=/path
#   -DFTSENSOR_ROOT=/path
#   -DSERIAL_ROOT=/path
#   -DHAND_ROOT=/path
set(EXPLICIT_CPP_ROOT "" CACHE PATH "Root of Explicit-cpp install/build")
set(FRI_ROOT          "" CACHE PATH "Root of FRI client install/build")
set(SERIAL_ROOT       "" CACHE PATH "Root of libSerial install (optional)")

# Helper to add an imported lib when no find_package config exists
function(_add_local_lib tgt libname rootvar)
  if(NOT "${${rootvar}}" STREQUAL "")
    find_library(${tgt}_LIB
      NAMES ${libname}
      HINTS "${${rootvar}}"
      PATH_SUFFIXES lib lib64)
    find_path(${tgt}_INC
      NAMES .
      HINTS "${${rootvar}}"
      PATH_SUFFIXES include)
    if(${tgt}_LIB)
      add_library(${tgt} UNKNOWN IMPORTED)
      set_target_properties(${tgt} PROPERTIES IMPORTED_LOCATION "${${tgt}_LIB}")
      if(${tgt}_INC)
        set_target_properties(${tgt} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${${tgt}_INC}")
      endif()
      target_link_libraries(${PROJECT_NAME} PRIVATE ${tgt})
    endif()
  endif()
endfunction()

# Core robot libs (adjust names if your actual files differ)
# e.g., libExplicit-cpp.so, libFRIClient.a
_add_local_lib(EXPLICIT_CPP  "Explicit-cpp" EXPLICIT_CPP_ROOT)
_add_local_lib(FRI_CLIENT    "FRIClient"    FRI_ROOT)

# ---- Warnings
if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Output dirs (multi-config friendly)
set_target_properties(${PROJECT_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
)

# ---- Install
install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})