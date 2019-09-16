#ifndef GRAPHNN_CLASIFIER_HPP
#define GRAPHNN_CLASIFIER_HPP

#include "GraphNN.hpp"
#include "GraphNNLearnableParameter.hpp"
#include "DatasetManager.hpp"
#include <Eigen/Core>
#include <cmath>

class GraphNNClassifier {
    public:
        GraphNNClassifier(uint32_t T) : T_(T) {}

        bool Classify(const GraphNN &nn, const GraphNNLearnableParameter &parameter) const {
            double p = GraphNNClassifier::Sigmoid(ComputeS(nn, parameter));
            return p > 0.5;
        }

        // Compute Average Loss
        template <typename D>
        double AverageLoss(const DatasetManager<D> &manager, const GraphNNLearnableParameter &parameter) const {
            double error = 0.0;
            for (auto data: manager) {
                error += Loss(data, parameter);
            }
            return error / manager.size();
        }

        // Compute LOSS
        double Loss(const Data &data, const GraphNNLearnableParameter &parameter) const {
            const auto& nn = data.GetInput();
            auto answer = data.GetLabel();

            double s = ComputeS(nn, parameter);
            double logOnePlusExpS = s > 400 ? s : std::log(1.0+std::exp(s));
            double logOnePlusInvExpS = s < -400 ? -s : std::log(1.0+std::exp(-s));

            return answer ? logOnePlusInvExpS : logOnePlusExpS;
        }

    private:
        static double Sigmoid(double x) {
            return 1.0 / (std::exp(-x) + 1.0);
        }

        double ComputeS(const GraphNN &nn, const GraphNNLearnableParameter &parameter) const {
            return parameter.GetA().dot(nn.ComputeOutput(T_, parameter.GetWeight())) + parameter.GetBias();
        }

        const uint32_t T_;
};

#endif