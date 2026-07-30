cmake_minimum_required(VERSION 3.16)
project(fer_ncnn_edge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FER_HAS_NCNN "Link against ncnn" OFF)
set(NCNN_DIR "" CACHE PATH "Path to ncnn install prefix (lib/cmake/ncnn)")
set(FER_NUM_CLASSES "7" CACHE STRING "FER classifier logits (7 ExpW / 8 FERPlus)")

add_executable(fer_ncnn_bench
  src/main.cpp
  src/fer_runner.cpp
  src/fer_timing.cpp
)
target_include_directories(fer_ncnn_bench PRIVATE include)

if(MSVC)
  target_compile_options(fer_ncnn_bench PRIVATE /utf-8)
endif()

if(FER_HAS_NCNN)
  if(NCNN_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NCNN_DIR}")
  endif()
  find_package(ncnn REQUIRED)
  target_compile_definitions(fer_ncnn_bench PRIVATE FER_HAS_NCNN=1 FER_NUM_CLASSES=${FER_NUM_CLASSES})
  target_link_libraries(fer_ncnn_bench PRIVATE ncnn)
  message(STATUS "fer_ncnn_bench: NCNN enabled FER_NUM_CLASSES=${FER_NUM_CLASSES}")
else()
  message(STATUS "fer_ncnn_bench: stub backend (protocol bring-up)")
endif()

if(FER_HAS_NCNN)
  add_executable(fer_op_bench src/fer_op_bench.cpp)
  target_link_libraries(fer_op_bench PRIVATE ncnn)
  if(MSVC)
    target_compile_options(fer_op_bench PRIVATE /utf-8)
  endif()
  install(TARGETS fer_op_bench RUNTIME DESTINATION bin)

  add_executable(fer_cpu_probe src/fer_cpu_probe.cpp)
  target_link_libraries(fer_cpu_probe PRIVATE ncnn)
  if(MSVC)
    target_compile_options(fer_cpu_probe PRIVATE /utf-8)
  endif()
  install(TARGETS fer_cpu_probe RUNTIME DESTINATION bin)

  add_executable(fer_ncnn_acc src/fer_ncnn_acc.cpp)
  target_link_libraries(fer_ncnn_acc PRIVATE ncnn)
  if(MSVC)
    target_compile_options(fer_ncnn_acc PRIVATE /utf-8)
  endif()
  install(TARGETS fer_ncnn_acc RUNTIME DESTINATION bin)

  add_executable(fer_early_exit src/fer_early_exit.cpp)
  target_link_libraries(fer_early_exit PRIVATE ncnn)
  if(MSVC)
    target_compile_options(fer_early_exit PRIVATE /utf-8)
  endif()
  install(TARGETS fer_early_exit RUNTIME DESTINATION bin)
endif()

install(TARGETS fer_ncnn_bench RUNTIME DESTINATION bin)
