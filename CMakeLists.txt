cmake_minimum_required(VERSION 3.0...3.25)
project(
	LoreSoftwareRenderer
	VERSION 0.1.0
	LANGUAGES CXX C
)

add_subdirectory(thirdparties/SDL)
set(SDL2IMAGE_VENDORED ON)
set(SDL2IMAGE_PNG ON)
set(SDL2IMAGE_AVIF OFF)
set(SDL2IMAGE_JPG ON)
set(SDL2IMAGE_TIF OFF)
set(SDL2IMAGE_WEBP OFF)
add_subdirectory(thirdparties/SDL_image)

add_executable(App
	main.cpp
)

file(GLOB SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

target_sources(App PRIVATE ${SRC_FILES})
target_include_directories(App PRIVATE src/include thirdparties/SDL/include thirdparties/SDL_image/include)
target_link_libraries(App PRIVATE SDL2 SDL2_image)
if (WIN32)
  target_link_libraries(
    App PRIVATE SDL2main
  )
endif()
target_include_directories(App PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/eigen)

set_target_properties(App PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
	COMPILE_WARNING_AS_ERROR ON
)

if (MSVC)
	target_compile_options(App PRIVATE /W4)
else()
	target_compile_options(App PRIVATE -Wall -Wextra -pedantic)
endif()

target_compile_definitions(App PRIVATE
    RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources"
)

target_compile_options(App PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4127 /wd4201 /wd4100 /wd4189 /wd4244>
)

add_custom_command(
  TARGET App POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:SDL2>"
    "$<TARGET_FILE:SDL2_image>"
    "$<TARGET_FILE_DIR:App>"
  VERBATIM
)