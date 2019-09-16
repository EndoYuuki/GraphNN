#ifndef VERTEXID_HPP
#define VERTEXID_HPP

class VertexID {
    public:
        VertexID(int id) 
            : id_(id) {
        }
        int GetID() const {
            return id_;
        }
        bool operator<(const VertexID &id) const {
            return id_ < id.id_;
        }
        bool operator==(const VertexID &id) const {
            return id_ == id.id_;
        }
    private:
        int id_;
};

#endif