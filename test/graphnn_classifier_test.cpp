#include <gtest/gtest.h>
#include <Eigen/Core>
#include "GraphNNClassifier.hpp"
#include "DatasetManager.hpp"
#include "GraphNNLearnableParameter.hpp"
#include "FileGraphGenerator.hpp"

class GraphNNClassifierTest : public ::testing::Test {
    protected:
        GraphNNClassifierTest() :
            classifier(2)
        {}

        GraphNNClassifier classifier;
};

TEST_F(GraphNNClassifierTest, NormalTest) {
    GraphNNLearnableParameter parameter(Eigen::MatrixXd::Constant(4,4,1.0), 1.0, Eigen::VectorXd::Constant(4, 0.01));

    Data data1(FileGraphGenerator("../../test_data.dat", 4), false);
    // graphNN readout result is [68,68,68,68]^T
    // 68*4*0.01 + 1 = 3.72
    // log(1.0 + exp(3.72)) ≒ 3.74394498474
    EXPECT_NEAR(3.74394498474, classifier.Loss(data1, parameter), 10e-10);
    EXPECT_NEAR(3.74394498474, classifier.Loss(data1, parameter), 10e-10);
    EXPECT_NEAR(3.74394498474, classifier.Loss(data1, parameter), 10e-10);

    Data data2(FileGraphGenerator("../../test_data.dat", 4), true);
    // graphNN readout result is [68,68,68,68]^T
    // 68*4*0.01 + 1 = 3.72
    // log(1.0 + exp(-3.72)) ≒ 0.02394498474
    EXPECT_NEAR(0.02394498474, classifier.Loss(data2, parameter), 10e-10);
    EXPECT_NEAR(0.02394498474, classifier.Loss(data2, parameter), 10e-10);
    EXPECT_NEAR(0.02394498474, classifier.Loss(data2, parameter), 10e-10);
}

TEST_F(GraphNNClassifierTest, OverflowTest) {
    GraphNNLearnableParameter parameter1(Eigen::MatrixXd::Constant(4,4,1.0), 1.0, Eigen::VectorXd::Constant(4, 2.0));

    Data data1(FileGraphGenerator("../../test_data.dat", 4), false);
    // graphNN readout result is [68,68,68,68]^T
    // 68*2*4 + 1 = 545
    // log(1.0 + exp(545)) ≒ 545
    double loss1 = classifier.Loss(data1, parameter1);
    EXPECT_FALSE(loss1 != loss1);
    EXPECT_NEAR(545, classifier.Loss(data1, parameter1), 10e-20);

    Data data2(FileGraphGenerator("../../test_data.dat", 4), true);
    // graphNN readout result is [68,68,68,68]^T
    // 68*2*4 + 1 = 545
    // log(1.0 + exp(-545)) ≒ 0
    EXPECT_NEAR(0, classifier.Loss(data2, parameter1), 10e-20);

    GraphNNLearnableParameter parameter2(Eigen::MatrixXd::Constant(4,4,1.0), 1.0, Eigen::VectorXd::Constant(4, -2.0));

    // graphNN readout result is [68,68,68,68]^T
    // -68*2*4 + 1 = -543
    // log(1.0 + exp(-543)) ≒ 0
    EXPECT_NEAR(0, classifier.Loss(data1, parameter2), 10e-20);

    // graphNN readout result is [68,68,68,68]^T
    // -68*2*4 + 1 = -543
    // log(1.0 + exp(543)) ≒ 543
    double loss2 = classifier.Loss(data2, parameter2);
    EXPECT_FALSE(loss2 != loss2);
    EXPECT_NEAR(543, classifier.Loss(data2, parameter2), 10e-20);
}