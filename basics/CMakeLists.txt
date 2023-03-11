
add_executable(basics basics.cpp)
target_link_libraries(basics "${TORCH_LIBRARIES}")

add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression "${TORCH_LIBRARIES}")
