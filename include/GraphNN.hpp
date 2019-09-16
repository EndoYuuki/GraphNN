#ifndef GRAPHNN_HPP
#define GRAPHNN_HPP

#include "Graph.hpp"
#include "GraphGenerator.hpp"
#include "Edges.hpp"
#include "Vertecies.hpp"
#include "VertexID.hpp"

#include <Eigen/Core>

class GraphNN {
    private:
        using vector = Eigen::VectorXd;
        using matrix = Eigen::MatrixXd;
        using vertecies = SamePropertyVertecies<vector>;
        using edges = UndirectedAdjacentEdges<bool>;
        using generator = GraphGenerator<vertecies, edges>;

    public:
        using vertex_const_iterator = typename vertecies::const_iterator;

        GraphNN(const Graph<vertecies, edges> &graph) 
            : graph_(graph) {}

        GraphNN(Graph<vertecies, edges> &&graph) 
            : graph_(std::move(graph)) {}

        GraphNN(GraphGenerator<vertecies, edges> &&generator) 
            : graph_(std::move(generator)){}

        vector GetVertex(const VertexID &id) const {
            return graph_.GetVertex(id);
        }

        vertex_const_iterator GetVerteciesBegin() const {
            return graph_.GetVerteciesBegin();
        }

        vertex_const_iterator GetVerteciesEnd() const {
            return graph_.GetVerteciesEnd(); 
        }

        vector ComputeOutput(int T, matrix W) const {
            Graph<vertecies, edges> tmp = graph_;
            for (int i = 0; i < T; i++) {
                tmp.Aggregate<matrix, vector>(W, GraphNN::ReLU);
            }
            return tmp.ReadOut<vector>();
        }

        const Graph<vertecies, edges>& GetGraph() const {
            return graph_;
        }

    private:
        static vector ReLU(const vector &x) {
            vector y(x.size());
            for (int i = 0; i < x.size(); i++) {
                y[i] = x[i] > 0 ? x[i] : 0;
            }
            return y;
        }
        Graph<vertecies, edges> graph_;
};

#endif