#include "GraphNNClassifier.hpp"
#include "GraphNN.hpp"
#include "GraphNNLearner.hpp"
#include "GraphNNLearnableParameter.hpp"
#include "FileGraphGenerator.hpp"
#include "DatasetManager.hpp"
#include <Eigen/Core>
#include <random>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void RunAllTrainer(const GraphNNClassifier &classifier,
    const DatasetManager<GraphNN> &trainManager, const DatasetManager<GraphNN> &validManager,
    int iteration, int batchSize);
GraphNNLearnableParameter InitializedParameter();

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "usage: program iteration batch_size dataset_path output_path" << std::endl;
        return -1;
    }
    int iteration = std::stoi(argv[1]);
    int batchSize = std::stoi(argv[2]);
    fs::path datasetPath(argv[3]);
    fs::path resultPath(argv[4]);

    GraphNNClassifier classifier(2);
    std::shared_ptr<DatasetManager<Data>> trainManager;
    std::shared_ptr<DatasetManager<Data>> validManager;

    try {
        trainManager.reset(
            new DatasetManager<Data>(
                DatasetLoader::LoadGraphAndLabel((datasetPath / "train/train/").string(), 1200, 8)));
        validManager.reset(
            new DatasetManager<Data>(
                DatasetLoader::LoadGraphAndLabel((datasetPath / "train/valid/").string(), 800, 8, 1200)));
    } catch (std::string message) {
        std::cout << message << std::endl;
        exit(-1);
    }

    // RunAllTrainer(classifier, *trainManager, *validManager, iteration, batchSize);

    // train using adam
    std::shared_ptr<GraphNNLearner> learner(new GraphNNAdam(classifier));
    GraphNNLearnableParameter parameter = InitializedParameter();
    for (int i = 0; i < iteration; i++)
    {
        // train
        DatasetBatchManager trainBatchManager(*trainManager, batchSize);
        (*learner)(trainBatchManager, parameter);
        std::cout << "iteration:" << i << std::endl;
    }

    // test
    std::ofstream test((resultPath / "results/test.txt").string());
    if (test.fail()) {
        std::cout << "Failed to open : " << "test" << std::endl;
        exit(-1);
    }

    std::shared_ptr<DatasetManager<GraphNN>> testManager;

    try {
        testManager.reset(
            new DatasetManager<GraphNN>(
                DatasetLoader::LoadGraph((datasetPath / "test/").string(), 1200, 8)));
    } catch (std::string message) {
        std::cout << message << std::endl;
        exit(-1);
    }

    for (auto itr = testManager->begin(); itr != testManager->end(); ++itr) {
        test << classifier.Classify(*itr, parameter) << std::endl;
    }

    return 0;
}

void RunAllTrainer(
    const GraphNNClassifier &classifier,
    const DatasetManager<Data> &trainManager,
    const DatasetManager<Data> &validManager,
    int iteration, int batchSize) {
    std::vector<std::pair<GraphNNLearner*, std::string>> learner = {
        std::make_pair(
            new GraphNNAdam(classifier),
            "../results/train_and_valid_adam.csv")};

    for (auto itr : learner) {
        GraphNNLearnableParameter parameter = InitializedParameter();
        std::ofstream result(itr.second);
        if (result.fail()) {
            std::cout << "Failed to open : " << "result" << std::endl;
            exit(-1);
        }

        for (int i = 0; i < iteration; i++) {
            // train
            DatasetBatchManager<Data> trainBatchManager(trainManager, batchSize);

            // valid
            result << classifier.AverageLoss(trainManager, parameter) << ","
                << classifier.AverageLoss(validManager, parameter) << std::endl;

            (*itr.first)(trainBatchManager, parameter);
        }
    }
}

GraphNNLearnableParameter InitializedParameter() {
    // init parameters
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> dist(0.0, 0.4);
    Eigen::VectorXd initA(8);
    Eigen::MatrixXd initW(8,8);
    for (int i = 0; i < 8; i++) {
        initA(i) = dist(engine);
        for (int j = 0; j < 8; j++) {
            initW(i,j) = dist(engine);
        }
    }
    return GraphNNLearnableParameter(initW, 0.0, initA);
}