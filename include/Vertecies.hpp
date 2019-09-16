#ifndef SAMEPROPERTYVERTECIES_HPP
#define SAMEPROPERTYVERTECIES_HPP

#include "VertexID.hpp"
#include <map>

template <class VProperty>
class SamePropertyVertecies {
    public:
        using property = VProperty;
        using map_container = std::map<VertexID, VProperty>;
        using const_iterator = typename map_container::const_iterator;

        SamePropertyVertecies(){}

        void Add(VertexID id, VProperty property) {
            vertecies_[id] = property;
        }

        const_iterator begin() const {
            return vertecies_.begin();
        }

        const_iterator end() const {
            return vertecies_.end();
        }

        property& operator[](const VertexID &id) {
            return vertecies_[id];
        }

        const property& at(const VertexID &id) const {
            return vertecies_.at(id);
        }

    private:
        map_container vertecies_;
};

#include <Eigen/Core>
template class SamePropertyVertecies<Eigen::VectorXd>;

#endif