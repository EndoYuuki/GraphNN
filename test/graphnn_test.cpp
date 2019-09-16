#include <gtest/gtest.h>

#include <vector>
#include "Graph.hpp"
#include "GraphGenerator.hpp"
#include "GraphNN.hpp"
#include "FileGraphGenerator.hpp"

#include <Eigen/Core>

using Eigen::MatrixXd;

class GraphNNTest : public ::testing::Test {
    protected:
        GraphNNTest() :
            graphNN(FileGraphGenerator("../../test_data.dat", 4)),
            weight(MatrixXd::Constant(4, 4, 1.0)){
        }
        GraphNN graphNN;
        MatrixXd weight;
};

TEST_F(GraphNNTest, InitialTest) {
    for (auto itr = graphNN.GetVerteciesBegin(); itr != graphNN.GetVerteciesEnd(); ++itr) {
        EXPECT_TRUE(itr->second[0] == 1.0);
        EXPECT_TRUE(itr->second[1] == 0.0);
        EXPECT_TRUE(itr->second[2] == 0.0);
        EXPECT_TRUE(itr->second[3] == 0.0);
    }
}
TEST_F(GraphNNTest, ReadOut) {
    // 0,1: (1,0,0,0)->(2,0,0,0)->(2,2,2,2)
    //      (2,2,2,2)->(5,5,5,5)->(20,20,20,20)
    //      (20,20,20,20)->(48,48,48,48)->(192,192,192,192)
    // 2: (1,0,0,0)->(3,0,0,0)->(3,3,3,3)
    //    (3,3,3,3)->(7,7,7,7)->(28,28,28,28)
    //    (28,28,28,28)->(68,68,68,68)->(272,272,272,272)
    EXPECT_TRUE((graphNN.ComputeOutput(3, weight).array() == 656).all());
}