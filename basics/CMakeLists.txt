

add_executable(basics basics.cpp)
target_link_libraries(basics "${TORCH_LIBRARIES}")

add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression "${TORCH_LIBRARIES}")

add_executable(logistic_regression logistic_regression.cpp)
target_link_libraries(logistic_regression "${TORCH_LIBRARIES}")
if(DOWNLOAD_DATASETS)
    add_dependencies(logistic_regression mnist)
endif()

add_executable(feed_forward_neural_network)
target_sources(
    feed_forward_neural_network PRIVATE
    neural_net.h
    neural_net.cpp
    feed_forward_neural_network.cpp
)
target_link_libraries(feed_forward_neural_network "${TORCH_LIBRARIES}")
if(DOWNLOAD_DATASETS)
    add_dependencies(logistic_regression mnist)
endif()
