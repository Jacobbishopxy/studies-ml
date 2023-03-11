
add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")

add_executable(introduction introduction.cpp)
target_link_libraries(introduction "${TORCH_LIBRARIES}")

add_executable(using_cpp_frontend using_cpp_frontend.cpp)
target_link_libraries(using_cpp_frontend "${TORCH_LIBRARIES}")

