cmake_minimum_required(VERSION 3.16)

project(RAGCLI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(third_party/CLI11)
add_subdirectory(third_party/json)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/Config.cpp
    src/Logger.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CLI11::CLI11
    nlohmann_json::nlohmann_json
)