# Copyright (c) Meta Platforms, Inc. and affiliates.

file(
    GLOB json_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.c")
list(FILTER json_sources EXCLUDE REGEX "${CMAKE_CURRENT_LIST_DIR}/tests/.*$")
file(
    GLOB json_parser_headers
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.h")

add_library(json_parser ${json_sources})
target_include_directories(json_parser PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR})
target_link_libraries(json_parser
    openzl
    shared_components
)

add_dependencies(json_parser openzl shared_components)
apply_openzl_compile_options_to_target(json_parser)
