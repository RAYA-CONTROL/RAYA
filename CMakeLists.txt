cmake_minimum_required(VERSION 3.20)
project(RAYA VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Keep operation rounding consistent with the frozen experiment protocol.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-ffp-contract=off)
endif()
add_library(raya_solver STATIC
  third_party/tinympc/src/tinympc/admm.cpp
  third_party/tinympc/src/tinympc/tiny_api.cpp
  third_party/tinympc/src/tinympc/rho_benchmark.cpp)
target_include_directories(raya_solver PUBLIC third_party/tinympc/src)
target_include_directories(raya_solver SYSTEM PUBLIC third_party/eigen)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # TinyMPC's existing C-linkage API returns Eigen objects in a few functions.
  target_compile_options(raya_solver PUBLIC -Wno-return-type-c-linkage)
endif()
foreach(platform quadrotor f1tenth)
  add_executable(raya_${platform} examples/${platform}/main.cpp)
  target_link_libraries(raya_${platform} PRIVATE raya_solver)
endforeach()
include(CTest)
if(BUILD_TESTING)
  add_executable(raya_sampling_check tests/check_sampling.cpp)
  add_test(NAME raya_sampling
    COMMAND raya_sampling_check ${CMAKE_CURRENT_SOURCE_DIR}/tests/sampling_vectors.csv)
  find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
  add_test(NAME raya_closed_loop
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_simulations.py
      --quadrotor $<TARGET_FILE:raya_quadrotor>
      --f1tenth $<TARGET_FILE:raya_f1tenth>)
  set_tests_properties(raya_closed_loop PROPERTIES TIMEOUT 180)
endif()
