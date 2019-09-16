cmake_minimum_required(VERSION 2.8.2)

find_package(GTest REQUIRED)

add_executable(graph_test graph_test.cpp)
target_link_libraries(graph_test gtest gmock_main)
add_test(NAME graph COMMAND $<TARGET_FILE:graph_test>)
# run with: ctest -L lib
set_property(TEST graph PROPERTY LABELS lib graph)

add_executable(graphnn_test graphnn_test.cpp)
target_link_libraries(graphnn_test gtest gmock_main)
add_test(NAME graphnn COMMAND $<TARGET_FILE:graphnn_test>)
# run with: ctest -L lib
set_property(TEST graphnn PROPERTY LABELS lib graphnn)

add_executable(dataset_test dataset_test.cpp)
target_link_libraries(dataset_test gtest gmock_main)
add_test(NAME dataset COMMAND $<TARGET_FILE:dataset_test>)
# run with: ctest -L lib
set_property(TEST dataset PROPERTY LABELS lib dataset)

add_executable(dataset_batch_test dataset_batch_test.cpp)
target_link_libraries(dataset_batch_test gtest gmock_main)
add_test(NAME dataset_batch COMMAND $<TARGET_FILE:dataset_batch_test>)
# run with: ctest -L lib
set_property(TEST dataset_batch PROPERTY LABELS lib dataset_batch)

add_executable(graphnn_classifier_test graphnn_classifier_test.cpp)
target_link_libraries(graphnn_classifier_test gtest gmock_main)
add_test(NAME graphnn_classifier COMMAND $<TARGET_FILE:graphnn_classifier_test>)
# run with: ctest -L lib
set_property(TEST graphnn_classifier PROPERTY LABELS lib graphnn_classifier)