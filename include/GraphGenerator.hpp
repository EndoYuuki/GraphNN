#ifndef GRAPHGENERATOR_HPP
#define GRAPHGENERATOR_HPP

#include "Graph.hpp"

template <class Vertecies, class Edges>
class GraphGenerator {
    public:
        GraphGenerator() {}
        virtual ~GraphGenerator() {}
        virtual void Initialize() = 0;
        virtual Edges CreateEdges() const = 0;
        virtual Vertecies CreateVertecies() const = 0;
        virtual void Finalize() = 0;

    private:
};

#endif