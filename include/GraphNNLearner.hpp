#ifndef GRAPHNN_LEARNER_HPP
#define GRAPHNN_LEARNER_HPP

#include "GraphNNClassifier.hpp"
#include "GraphNNLearnableParameter.hpp"

#include "DatasetManager.hpp"
#include <future>

/*
    This class manages a graph neural network to learn.
    Learning method is defined in derived class.
*/
class GraphNNLearner {
    public:
        GraphNNLearner(GraphNNClassifier classifier, double epsilon = 0.001)
            :   classifier_(classifier), epsilon_(epsilon){}

        virtual ~GraphNNLearner() {}

        // gradient desent abstraction which suggests parameter updates.
        virtual void operator() (const DatasetBatchManager<Data> &manager, GraphNNLearnableParameter &parameter) const = 0;

    protected:
        // Compute a mean of a gradient to a dataset batch manager.
        // This method uses std::thread for a rapid computation.
        Eigen::VectorXd ComputeGrad(const DatasetBatchManager<Data>::batch_type &batch, const GraphNNLearnableParameter &param) const {
            std::vector<std::future<Eigen::VectorXd>> results;

            for (auto data: batch) {
                // because of right reference, use emplace back for efficiency.
                results.emplace_back(
                    std::async(
                        std::launch::async,
                        [this](Data d, GraphNNLearnableParameter p){
                            return ComputeNumericGrad(d, p);
                        }, data, param));
            }

            Eigen::VectorXd meanGrad = Eigen::VectorXd::Zero(param.Size());

            for (auto &result: results) {
                meanGrad += result.get();
            }

            return meanGrad / batch.size();
        }

    private:
        // compute numeric gradient for a target.
        Eigen::VectorXd ComputeNumericGrad(const Data &data, const GraphNNLearnableParameter &parameter) const {
            Eigen::VectorXd result(parameter.Size());
            auto loss = classifier_.Loss(data, parameter);
            for (int i = 0; i < parameter.Size(); i++) {
                result[i] = (classifier_.Loss(data, parameter.DeltaChanged(i, epsilon_)) - loss)/epsilon_;
            }
            return result;
        }

    protected:
        GraphNNClassifier classifier_;
        const double epsilon_;
};

class GraphNNSGD : public GraphNNLearner {
    public:
        GraphNNSGD(GraphNNClassifier classifier, double epsilon = 0.001, double alpha = 0.0001)
            : GraphNNLearner(classifier, epsilon), alpha_(alpha) {
        }

        void operator() (const DatasetBatchManager<Data> &manager, GraphNNLearnableParameter &parameter) const {
            for (auto batch : manager) {
                parameter.GradDesent(
                    GraphNNLearner::ComputeGrad(batch, parameter) * alpha_);
            }
        }
    private:
        const double alpha_;
};

class GraphNNMomentumSGD : public GraphNNLearner {
    public:
        GraphNNMomentumSGD(GraphNNClassifier classifier, double epsilon = 0.001, double mu = 1.0, double alpha = 0.0001)
            : GraphNNLearner(classifier, epsilon), alpha_(alpha), mu_(mu) {
        }

        void operator() (const DatasetBatchManager<Data> &manager, GraphNNLearnableParameter &parameter) const {
            Eigen::VectorXd w = Eigen::VectorXd::Zero(parameter.Size());
            for (auto batch : manager) {
                auto grad = ComputeGrad(batch, parameter);
                parameter.GradDesent(grad * alpha_ - w * mu_);
                w = -grad * alpha_ + w * mu_;
            }
        }
    private:
        const double mu_;
        const double alpha_;
};

class GraphNNAdam : public GraphNNLearner {
    public:
        GraphNNAdam(GraphNNClassifier classifier, double epsilon = 0.001, double epsilon2 = 1e-8, double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999)
            : GraphNNLearner(classifier, epsilon), epsilon_(epsilon2), alpha_(alpha), beta1_(beta1), beta2_(beta2) {
        }

        void operator() (const DatasetBatchManager<Data> &manager, GraphNNLearnableParameter &parameter) const {
            int t = 1;
            Eigen::VectorXd m = Eigen::VectorXd::Zero(parameter.Size());
            Eigen::VectorXd v = Eigen::VectorXd::Zero(parameter.Size());
            for (auto batch : manager) {
                auto grad = ComputeGrad(batch, parameter);

                m = beta1_ * m + (1-beta1_) * grad;
                v = beta2_ * v + (1-beta2_) * grad.array().square().matrix();

                auto mhat = m / (1-std::pow(beta1_, t));
                auto vhat = v / (1-std::pow(beta2_, t));

                Eigen::VectorXd desent(parameter.Size());
                for (int i = 0; i < parameter.Size(); i++) {
                    desent[i] = alpha_ * mhat[i] / (std::sqrt(vhat[i]) + epsilon_);
                }
                parameter.GradDesent(desent);
                t++;
            }
        }
    private:
        const double alpha_;
        const double beta1_, beta2_;
        const double epsilon_;
};

#endif