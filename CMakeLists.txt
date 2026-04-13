cmake_minimum_required(VERSION 3.25)
project(DspHeterodyne VERSION 0.1.0
  DESCRIPTION "LFM Dechirp, NCO, MixDown/MixUp"
  LANGUAGES CXX HIP)

# ── Опции ──────────────────────────────────────────────────────────
option(DSP_HETERODYNE_BUILD_TESTS  "Build tests"            ON)
option(DSP_HETERODYNE_BUILD_PYTHON "Build Python bindings"  OFF)

# ── Зависимости ────────────────────────────────────────────────────
find_package(hip    REQUIRED)   # lowercase — Linux case-sensitive!
find_package(hipfft REQUIRED)

include(cmake/fetch_deps.cmake)
fetch_dsp_core()
fetch_dsp_spectrum()
fetch_dsp_signal_generators()

# ── Библиотека ─────────────────────────────────────────────────────
add_library(DspHeterodyne STATIC)
add_library(DspHeterodyne::DspHeterodyne ALIAS DspHeterodyne)

target_sources(DspHeterodyne PRIVATE
  src/heterodyne/src/heterodyne_dechirp.cpp
  src/heterodyne/src/heterodyne_processor_rocm.cpp
)

target_include_directories(DspHeterodyne
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    src/heterodyne/include
    kernels/
    src
)
target_link_libraries(DspHeterodyne
  PUBLIC DspCore::DspCore DspSpectrum::DspSpectrum DspSignalGenerators::DspSignalGenerators
)
target_compile_features(DspHeterodyne PUBLIC cxx_std_17)

# ── Git Version ────────────────────────────────────────────────────
find_package(Git QUIET)
if(NOT GIT_EXECUTABLE)
    set(GIT_EXECUTABLE git)
endif()

add_custom_target(git_version_DspHeterodyne ALL
    COMMAND ${CMAKE_COMMAND}
        -D SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -D BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
        -D MODULE_PREFIX=DSPHETERODYNE
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake
    COMMENT "[DspHeterodyne] Checking git version..."
    VERBATIM
)
add_dependencies(DspHeterodyne git_version_DspHeterodyne)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS DspHeterodyne EXPORT DspHeterodyneTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DspHeterodyne
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.json"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/DspHeterodyne
)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/DspHeterodyneConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DspHeterodyneConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspHeterodyne
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DspHeterodyneConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)
install(EXPORT DspHeterodyneTargets NAMESPACE DspHeterodyne::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspHeterodyne
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DspHeterodyneConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DspHeterodyneConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspHeterodyne
)

if(DSP_HETERODYNE_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(DSP_HETERODYNE_BUILD_PYTHON)
  add_subdirectory(python)
endif()
