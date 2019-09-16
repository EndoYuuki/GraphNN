#include <gtest/gtest.h>

#include "Graph.hpp"
#include "GraphGenerator.hpp"
#include "FileGraphGenerator.hpp"

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXd;

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

class EdgesTest : public ::testing::Test {
    protected:
        EdgesTest() {
        }
        UndirectedAdjacentEdges<bool> edges;
};

TEST_F(EdgesTest, FromMapInitialize) {
    UndirectedAdjacentEdges<bool>::map_type map;
    map[std::make_pair(0, 1)] = true;
    map[std::make_pair(0, 2)] = true;
    map[std::make_pair(2, 1)] = true;
    edges = UndirectedAdjacentEdges<bool>(map);
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(0), VertexID(1)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(0), VertexID(2)));
    EXPECT_FALSE(vector_finder(edges.GetAdjacentIDList(0), VertexID(0)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(1), VertexID(0)));
    EXPECT_FALSE(vector_finder(edges.GetAdjacentIDList(1), VertexID(1)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(1), VertexID(2)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(2), VertexID(1)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(2), VertexID(0)));
    EXPECT_FALSE(vector_finder(edges.GetAdjacentIDList(2), VertexID(2)));
}

TEST_F(EdgesTest, FromEmptyInitialize) {
    edges.Add(0, 1, true);
    edges.Add(0, 2, true);
    edges.Add(2, 1, true);
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(0), VertexID(1)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(0), VertexID(2)));
    EXPECT_FALSE(vector_finder(edges.GetAdjacentIDList(0), VertexID(0)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(1), VertexID(0)));
    EXPECT_FALSE(vector_finder(edges.GetAdjacentIDList(1), VertexID(1)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(1), VertexID(2)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(2), VertexID(1)));
    EXPECT_TRUE(vector_finder(edges.GetAdjacentIDList(2), VertexID(0)));
    EXPECT_FALSE(vector_finder(edges.GetAdjacentIDList(2), VertexID(2)));
}

class GraphTest : public ::testing::Test {
    protected:
        using GraphType = Graph<SamePropertyVertecies<Eigen::VectorXd>, UndirectedAdjacentEdges<bool>>;
        GraphTest() :
            graph(FileGraphGenerator("../../test_data.dat", 10)){
        }
        GraphType graph;
};

TEST_F(GraphTest, InitialTest) {
    EXPECT_TRUE(vector_finder(graph.GetAdjacentIDList(VertexID(0)), VertexID(2)));
    EXPECT_FALSE(vector_finder(graph.GetAdjacentIDList(VertexID(0)), VertexID(0)));
    EXPECT_FALSE(vector_finder(graph.GetAdjacentIDList(VertexID(0)), VertexID(1)));

    EXPECT_TRUE(vector_finder(graph.GetAdjacentIDList(VertexID(1)), VertexID(2)));
    EXPECT_FALSE(vector_finder(graph.GetAdjacentIDList(VertexID(1)), VertexID(0)));
    EXPECT_FALSE(vector_finder(graph.GetAdjacentIDList(VertexID(1)), VertexID(1)));

    EXPECT_TRUE(vector_finder(graph.GetAdjacentIDList(VertexID(2)), VertexID(0)));
    EXPECT_TRUE(vector_finder(graph.GetAdjacentIDList(VertexID(2)), VertexID(1)));
    EXPECT_FALSE(vector_finder(graph.GetAdjacentIDList(VertexID(2)), VertexID(2)));
}