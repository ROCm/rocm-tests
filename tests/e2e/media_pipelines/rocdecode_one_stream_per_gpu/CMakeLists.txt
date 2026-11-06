cmake_minimum_required(VERSION 3.21)

# Framework builds pass ROCM_TEST_CMAKE_MODULE_DIR and the compiler on the
# cmake command line. A standalone `cmake -S . -B build` keeps the old default:
# ROCM_PATH from the environment, or /opt/rocm, and ROCm clang++.
if(ROCM_TEST_CMAKE_MODULE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${ROCM_TEST_CMAKE_MODULE_DIR}")
    include(rocm_setup)
else()
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
    if(NOT CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER ${ROCM_PATH}/llvm/bin/clang++ CACHE FILEPATH "ROCm clang can parse the HIP host headers" FORCE)
    endif()
endif()

project(rocdecode_one_stream_per_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

find_package(hip REQUIRED)
find_library(ROCDECODE_LIBRARY NAMES rocdecode PATHS ${ROCM_PATH}/lib REQUIRED)

set(SAMPLE_STREAM ${CMAKE_CURRENT_SOURCE_DIR}/media/sample_stream.h264)
if(NOT EXISTS ${SAMPLE_STREAM})
    message(FATAL_ERROR "Missing H.264 elementary stream: ${SAMPLE_STREAM}")
endif()

add_library(decode_placement_common STATIC common/result_table.cpp)
target_include_directories(decode_placement_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_compile_definitions(decode_placement_common PUBLIC __HIP_PLATFORM_AMD__)
target_link_libraries(decode_placement_common PUBLIC hip::host)

function(add_decode_mode mode target)
    add_executable(${target} rocdecode_one_stream_per_gpu.cpp)
    target_link_libraries(${target} PRIVATE decode_placement_common ${ROCDECODE_LIBRARY})
    target_include_directories(${target} PRIVATE ${ROCM_PATH}/include)
    target_compile_definitions(${target} PRIVATE SAMPLE_STREAM_PATH="${SAMPLE_STREAM}" TEST_MODE_${mode})
    set_target_properties(${target} PROPERTIES BUILD_RPATH "${ROCM_PATH}/lib;${ROCM_PATH}/lib/rocm_sysdeps/lib;${ROCM_PATH}/llvm/lib")
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 2 TIMEOUT 600)
endfunction()

set(XCD_BURN ${CMAKE_CURRENT_BINARY_DIR}/xcd_burn)
set(XCD_BURN_FLAGS -O2)
if(GPU_ARCH)
    list(APPEND XCD_BURN_FLAGS --offload-arch=${GPU_ARCH})
endif()
add_custom_command(
    OUTPUT ${XCD_BURN}
    COMMAND ${ROCM_PATH}/bin/hipcc ${XCD_BURN_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/xcd_burn.cpp -o ${XCD_BURN}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/xcd_burn.cpp
    VERBATIM
)
add_custom_target(build_xcd_burn ALL DEPENDS ${XCD_BURN})

add_decode_mode(SPX rocdecode_spx_one_stream_per_card)
add_decode_mode(DPX rocdecode_dpx_one_stream_per_half)
add_decode_mode(CPX rocdecode_cpx_one_stream_per_slice)
add_dependencies(rocdecode_spx_one_stream_per_card build_xcd_burn)
add_dependencies(rocdecode_dpx_one_stream_per_half build_xcd_burn)
add_dependencies(rocdecode_cpx_one_stream_per_slice build_xcd_burn)
