
set(CNN convolutional-neural-network)

add_executable(${CNN})
target_sources(
    ${CNN} PRIVATE
    convolutional_neural_network/convnet.h
    convolutional_neural_network/imagefolder_dataset.h
    convolutional_neural_network/convnet.cpp
    convolutional_neural_network/imagefolder_dataset.cpp
    convolutional_neural_network/convolutional_neural_network.cpp
)
target_include_directories(${CNN} PRIVATE convolutional_neural_network)
target_link_libraries(${CNN} "${TORCH_LIBRARIES}" image-io)

if(DOWNLOAD_DATASETS)
    add_dependencies(${CNN} imagenette)
endif()


set(DRN deep-residual-network)

add_executable(${DRN})
target_sources(
    ${DRN} PRIVATE
    deep_residual_network/cifar10.h
    deep_residual_network/residual_block.h
    deep_residual_network/resnet.h
    deep_residual_network/transform.h
    deep_residual_network/deep_residual_network.cpp
)
target_include_directories(${DRN} PRIVATE deep_residual_network)
target_link_libraries(${DRN} "${TORCH_LIBRARIES}")

if(DOWNLOAD_DATASETS)
    add_dependencies(${DRN} cifar10)
endif()
