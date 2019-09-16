#ifndef GRAPHNN_LEARNABLE_PARAMETER_HPP
#define GRAPHNN_LEARNABLE_PARAMETER_HPP

#include <Eigen/Core>

class GraphNNLearnableParameter {
    public: 
        GraphNNLearnableParameter(Eigen::MatrixXd weight, double bias, Eigen::VectorXd A) 
            : weight_(weight), A_(A), bias_(bias) {
        }

        const Eigen::MatrixXd& GetWeight() const {
            return weight_;
        }
        const Eigen::VectorXd& GetA() const {
            return A_;
        }
        const double GetBias() const {
            return bias_;
        }

        GraphNNLearnableParameter DeltaChanged(int i, double delta) const {
            GraphNNLearnableParameter result = (*this);
            result[i] += delta;//result[i] * delta;
            return result;
        }

        void GradDesent(const Eigen::VectorXd &grad) {
            for (int i = 0; i < Size(); i++) {
                operator[](i) -= grad[i];
            }
        }

        const double& operator[] (int i) const {
            int limit = weight_.rows()*weight_.cols();
            if (i < limit) {
                return *(weight_.data() + i);
            }
            limit += A_.rows();
            if (i < limit) {
                return *(A_.data()+(i-weight_.rows() * weight_.cols()));
            }
            return bias_;
        }

        double& operator[] (int i) {
            int limit = weight_.rows()*weight_.cols();
            if (i < limit) {
                return *(weight_.data() + i);
            }
            limit += A_.rows();
            if (i < limit) {
                return *(A_.data()+(i-weight_.rows() * weight_.cols()));
            }
            return bias_;
        }

        int Size() const {
            return weight_.rows() * weight_.cols() + A_.rows() + 1;
        }

    private:
        Eigen::MatrixXd weight_;
        Eigen::VectorXd A_;
        double bias_;
};

#endif