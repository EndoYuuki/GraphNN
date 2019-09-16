#ifndef GRAPHNN_CLASSIFIER_MODEL_HPP
#define GRAPHNN_CLASSIFIER_MODEL_HPP

#include "GraphNNClassifier.hpp"
#include "GraphNNLearnableParameter.hpp"

class GraphNNClassifierModel {
    public:
        GraphNNClassifierModel(GraphNNClassifier classifier, GraphNNLearnableParameter parameter) 
            : classifier_(classifier), parameter_(parameter) {}

        void SetParameter(GraphNNLearnableParameter parameter) {
            parameter_ = parameter;
        }

    private:
        GraphNNClassifier classifier_;
        GraphNNLearnableParameter parameter_;
};

#endif