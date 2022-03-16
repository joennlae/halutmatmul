cmake_minimum_required(VERSION 3.5.1)

# project information
project(maddness)

# Compile options
add_compile_options(-std=c++14)

set(CMAKE_CXX_FLAGS "-O3 -march=native -mavx -ffast-math")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../out")

message(STATUS "CLANG_TIDY : ${CLANG_TIDY}")
message(STATUS "ORIGIN : ${ORIGIN}")

include_directories(
    lib/eigen/
    src/include
    test/include/
)
# add host lib path
# link_directories(
# )


if ("${CLANG_TIDY}" STREQUAL "0")
    message(STATUS "Building normal")
    # Header path
else ()
    message(STATUS "TODO implement clang-tidy")
    set(CMAKE_CXX_CLANG_TIDY clang-tidy-12.0.1;) # -fix-errors; or -fix;
endif()

set (COMMON_SOURCES
  src/mithral.cpp
  src/avx_utils.cpp
)
set(COMMON_LIBRARIES
  # none at the moment
)

add_executable(maddness
  src/main.cpp
  ${COMMON_SOURCES}
)

# Clang specific stdlib
if ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()


target_link_libraries(maddness
  ${COMMON_LIBRARIES}
)

install(TARGETS maddness DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})


if ("${UNIT_TESTS}" STREQUAL "1")

  add_subdirectory(lib/Catch2)
  # https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#cmake-targets

  message(STATUS "Building unit tests")

  set( TEST_DIR test )
  # set( HEADER_DIR  test/external/ )

  # main for idiomatic test sources:

  set( SOURCES_IDIOMATIC_MAIN
    # main_test.cpp
  )

  # sources to combine with 000-CatchMain.cpp:

  set( SOURCES_IDIOMATIC_TESTS
    profile_amm.cpp
    profile_encode.cpp
    profile_lut_creation.cpp
    profile_multicodebook.cpp
    profile_scan.cpp
    test_mithral.cpp
    test_multicodebook.cpp
  )

  # check if all sources are listed, warn if not:

  set( SOURCES_ALL
      ${SOURCES_IDIOMATIC_MAIN}
      ${SOURCES_IDIOMATIC_TESTS}
  )

  foreach( name ${SOURCES_ALL} )
      list( APPEND SOURCES_ALL_PATH ${TEST_DIR}/${name} )
  endforeach()

  # CheckFileList( SOURCES_ALL_PATH ${TEST_DIR} )

  # create target names:

  string( REPLACE ".cpp" "" BASENAMES_IDIOMATIC_TESTS "${SOURCES_IDIOMATIC_TESTS}" )

  set( TARGETS_IDIOMATIC_TESTS ${BASENAMES_IDIOMATIC_TESTS} )

  set( TARGETS_ALL
      ${TARGETS_IDIOMATIC_TESTS} CatchMain
  )

  # define program targets:

  # add_library( CatchMain         OBJECT ${TEST_DIR}/${SOURCES_IDIOMATIC_MAIN} ${HEADER_DIR}/catch.hpp )

  foreach( name ${TARGETS_IDIOMATIC_TESTS} )
      # add_executable( ${name} ${COMMON_SOURCES} ${TEST_DIR}/${name}.cpp $<TARGET_OBJECTS:CatchMain> ${HEADER_DIR}/catch.hpp )
      add_executable( ${name} ${COMMON_SOURCES} ${TEST_DIR}/${name}.cpp)
      target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain)
  endforeach()

  # foreach( name ${TARGETS_ALL} )
  #     target_include_directories( ${name} )
# 
  #     set_property(TARGET ${name} PROPERTY CXX_STANDARD 14)
  #     set_property(TARGET ${name} PROPERTY CXX_EXTENSIONS OFF)
# 
  #     # Add desired warnings
  #     if ( CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU" )
  #         # target_compile_options( ${name}  PRIVATE -Wall -Wextra -Wunreachable-code )
  #     endif()
  #     # Clang specific warning go here
  #     if ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  #         # Actually keep these
  #         # target_compile_options( ${name}  PRIVATE -Wweak-vtables -Wexit-time-destructors -Wglobal-constructors -Wmissing-noreturn )
  #     endif()
  #     if ( CMAKE_CXX_COMPILER_ID MATCHES "MSVC" )
  #         # target_compile_options( ${name}  PRIVATE /W4 /w44265 /WX )
  #     endif()
  # endforeach()

endif()