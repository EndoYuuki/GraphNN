#ifndef FILEGRAPHGENERATOR_HPP
#define FILEGRAPHGENERATOR_HPP

#include "GraphGenerator.hpp"
#include "VertexID.hpp"
#include "Vertecies.hpp"
#include "Edges.hpp"
#include <Eigen/Core>
#include <fstream>
#include <iostream>

/*
    This class generate graph from a file.
    The file format is following:

    numberOfVerticies
    0 1 1 0
    1 0 0 1
    1 0 1 1
    0 1 1 0

    The first line of the file shows number of verticies in a graph.
    After second lines, each row and column suggests vertex and 
    the value means "there is a connection between these vertecies".
    Since this graph is defined as UndirectedAdjacentGraph, this matrix should be symnetric.
*/
class FileGraphGenerator: public GraphGenerator<
        SamePropertyVertecies<Eigen::VectorXd>,
        UndirectedAdjacentEdges<bool>> {
    public:
        using edges = UndirectedAdjacentEdges<bool>;
        using vertecies = SamePropertyVertecies<Eigen::VectorXd>;
        using vertex_property = vertecies::property;

        FileGraphGenerator(const std::string &file, int dim) : filename_(file), dim_(dim) {}

        void Initialize() {
            std::ifstream stream(filename_);
            if (stream.fail()) {
                throw "Failed to open: " + filename_;
            }
            // number of vertecies
            stream >> numOfVertecies_;

            // adjacency lists
            for (int i = 0; i < numOfVertecies_; i++) {
                for (int j = 0; j < numOfVertecies_; j++) {
                    int m;
                    stream >> m;
                    if (j >= i) continue;
                    if (m == 1) {
                        table_[std::make_pair(VertexID(i), VertexID(j))] = true;
                    }
                }
            }

            stream.close();
        }
        void Finalize() {
            // nothing to do
        }
        edges CreateEdges() const {
            return edges(table_);
        }
        vertecies CreateVertecies() const {
            vertecies v;
            for (int i = 0; i < numOfVertecies_; i++) {
                vertex_property vp = vertex_property::Zero(dim_);

                // temporally, set the value to each vertex, since graph should has value in my scheme.
                // should be improved...
                vp[0] = 1.0;
                v.Add(i, vp);
            }
            return v;
        }

    private:
        std::string filename_;
        std::map<std::pair<VertexID, VertexID>, bool> table_;
        int dim_;
        int numOfVertecies_;
};

#endif