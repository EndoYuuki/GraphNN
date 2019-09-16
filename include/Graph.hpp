#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "GraphGenerator.hpp"
#include "VertexID.hpp"
#include <map>

template <class Vertecies, class Edges>
class Graph {
    public:
        using key_list = typename Edges::key_list;
        using edge_property = typename Edges::property;
        using vertex_property = typename Vertecies::property;
        using vertex_const_iterator = typename Vertecies::const_iterator;

        Graph(GraphGenerator<Vertecies, Edges> &&generator) {
            generator.Initialize();
            vertecies_ = generator.CreateVertecies();
            edges_ = generator.CreateEdges();
            generator.Finalize();
        }

        void AddVertex(VertexID id, vertex_property property) {
            vertecies_.Add(id, property);
        }

        vertex_property GetVertex(const VertexID &id) const {
            return vertecies_.at(id);
        }

        void SetVertex(const VertexID &id, vertex_property property) {
            vertecies_[id] = property;
        }

        void AddEdge(edge_property property, const VertexID &id1, const VertexID &id2) {
            edges_.Add(id1, id2, property);
        }

        key_list GetAdjacentIDList(const VertexID &id) const {
            return edges_.GetAdjacentIDList(id);
        }

        vertex_const_iterator GetVerteciesBegin() const {
            return vertecies_.begin();
        }

        vertex_const_iterator GetVerteciesEnd() const {
            return vertecies_.end();
        }
        
        template <class matrix, class vector, class acctivate>
        void Aggregate(const matrix &weight, acctivate fn) {
            std::map<VertexID, vector> tmp;
            for (auto itr = GetVerteciesBegin(); itr != GetVerteciesEnd(); ++itr) {
                // first step
                auto idList = GetAdjacentIDList(itr->first);
                auto a = GetVertex(itr->first);
                for (auto idItr = idList.begin(); idItr != idList.end(); ++idItr) {
                    a += GetVertex(*idItr);
                }
                // second step
                tmp[itr->first] = fn(weight * a);
            }
            for (auto t: tmp) {
                SetVertex(t.first, t.second);
            }
        }

        template <class vector>
        vector ReadOut() const {
            vector v = GetVerteciesBegin()->second;
            for (auto itr = std::next(GetVerteciesBegin(), 1); itr != GetVerteciesEnd(); ++itr) {
                v += itr->second;
            }
            return v;
        }

    private:
        Vertecies vertecies_;
        Edges edges_;
};

#endif