#ifndef GRAPHOPERATOR_HPP
#define GRAPHOPERATOR_HPP

#include "Graph.hpp"

template <class Vertecies, class Edges>
class GraphOperator {
    public:
        using vertex_const_iterator = typename Graph<Vertecies, Edges>::vertex_const_iterator;

        GraphOperator(Graph graph) : graph_(graph){
        }

        vertex_const_iterator GetAdjsentVertecies(const VertexID &id) const {
        }

    private:
        Graph graph_;
};

#endif