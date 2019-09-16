#include <gtest/gtest.h>

#include "DatasetManager.hpp"

// check wheather vector contains the value or not
template <class T>
bool vector_finder(std::vector<T> vec, T number) {
    auto itr = std::find(vec.begin(), vec.end(), number);
    size_t index = std::distance(vec.begin(), itr);
    if (index != vec.size()) { // 発見できたとき
        return true;
    }
    else { // 発見できなかったとき
        return false;
    }
}

TEST(DatasetManagerTest, FailedReadTest) {
    ASSERT_ANY_THROW(DatasetLoader::LoadGraphAndLabel("../../datasets/train/train/", 1201, 8));
}

TEST(DatasetManagerTest, SuccessReadTest) {
    DatasetManager<Data> manager(
        DatasetLoader::LoadGraphAndLabel("../../datasets/train/train/", 1200, 8));
    Data data = manager[2];

    EXPECT_TRUE(data.GetLabel());
    auto graph = data.GetInput().GetGraph();

    for (auto vItr = graph.GetVerteciesBegin(); vItr != graph.GetVerteciesEnd(); ++vItr) {
        EXPECT_TRUE(vItr->second[0] == 1.0);
        for (int i = 1; i < 8; i++) {
            EXPECT_TRUE(vItr->second[i] == 0.0);
        }
    }

    EXPECT_TRUE(vector_finder(graph.GetAdjacentIDList(VertexID(0)), VertexID(5)));
    EXPECT_TRUE(vector_finder(graph.GetAdjacentIDList(VertexID(0)), VertexID(8)));
    EXPECT_FALSE(vector_finder(graph.GetAdjacentIDList(VertexID(0)), VertexID(1)));

    EXPECT_TRUE(vector_finder(graph.GetAdjacentIDList(VertexID(7)), VertexID(1)));
    EXPECT_TRUE(vector_finder(graph.GetAdjacentIDList(VertexID(7)), VertexID(8)));
    EXPECT_FALSE(vector_finder(graph.GetAdjacentIDList(VertexID(7)), VertexID(0)));
}