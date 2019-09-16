#ifndef EDGES_HPP
#define EDGES_HPP

#include "VertexID.hpp"

#include <vector>
#include <map>

template <class EProperty>
class UndirectedAdjacentEdges {
    public:
        using key_list = std::vector<VertexID>;
        using property = EProperty;
        using map_type = std::map<std::pair<VertexID, VertexID>, EProperty>;

        UndirectedAdjacentEdges() {}
        UndirectedAdjacentEdges(const map_type &table) : table_(table) {}
        UndirectedAdjacentEdges(map_type &&table) : table_(std::move(table)){}

        void Add(const VertexID &id1, const VertexID &id2, EProperty property) {
            table_[std::make_pair(id1, id2)] = property;
        }

        key_list GetAdjacentIDList(const VertexID &id) const {
            key_list adjacentIDList;
            for (auto t: table_) {
                auto &pair = t.first;
                if (pair.first == id) {
                    adjacentIDList.push_back(pair.second);
                }
                else if (pair.second == id) {
                    adjacentIDList.push_back(pair.first);
                }
            }
            return adjacentIDList;
        }

    private:
        map_type table_;
};

template class UndirectedAdjacentEdges<bool>;

#endif