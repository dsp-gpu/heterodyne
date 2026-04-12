# ── DspHeterodyne tests ────────────────────────────────────────────
add_executable(test_heterodyne_main main.cpp)
target_link_libraries(test_heterodyne_main PRIVATE
  DspHeterodyne::DspHeterodyne DspCore::TestUtils)
target_include_directories(test_heterodyne_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_heterodyne_main PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>)
