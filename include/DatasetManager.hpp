#ifndef DATASETMANAGER_HPP
#define DATASETMANAGER_HPP

#include "GraphNN.hpp"
#include "FileGraphGenerator.hpp"
#include <fstream>

class Data {
    public:
        Data(const GraphNN &input, bool label)
            : input_(input), label_(label) {}

        Data(GraphNN &&input, bool label)
            : input_(std::move(input)), label_(label) {}

        const GraphNN& GetInput() const {
            return input_;
        }
        bool GetLabel() const {
            return label_;
        }

    private:
        GraphNN input_;
        bool label_;
};

class LabelLoader {
    public:
        static bool load(const std::string &file) {
            std::ifstream ifs(file);
            if (ifs.fail()) {
                throw "Failed to open: " + file;
            }
            int i;
            ifs >> i;
            return i == 1;
        }
};

class DatasetLoader
{
    public:
        static std::vector<Data> LoadGraphAndLabel(const std::string &rootDir, int num, int dim, int first = 0)
        {
            std::vector<Data> dataList;
            for (int i = first; i < num + first; i++)
            {
                Data data(
                    FileGraphGenerator(rootDir + std::to_string(i) + "_graph.txt", dim),
                     LabelLoader::load(rootDir + std::to_string(i) + "_label.txt")
                );
                dataList.push_back(data);
            }
            return dataList;
        }

        static std::vector<GraphNN> LoadGraph(const std::string &rootDir, int num, int dim, int first = 0)
        {
            std::vector<GraphNN> graphList;
            for (int i = first; i < num + first; i++)
            {
                graphList.push_back(FileGraphGenerator(rootDir + std::to_string(i) + "_graph.txt", dim));
            }
            return graphList;
        }
};

template <typename D>
class DatasetManager {
    public:
        using const_iterator = typename std::vector<D>::const_iterator;
        using size_type = typename std::vector<D>::size_type;
        using const_reference = typename std::vector<D>::const_reference;

        DatasetManager(const std::vector<D> &dataList) :
            dataList_(dataList) {}

        const_iterator begin() const {
            return dataList_.begin();
        }

        const_iterator end() const {
            return dataList_.end();
        }

        size_type size() const {
            return dataList_.size();
        }

        const_reference operator[] (size_type idx) const {
            return dataList_[idx];
        }

    private:
        std::vector<D> dataList_;
};

template <typename D>
class DatasetBatchManager {
    public:
        using batch_type = std::vector<D>;
        using batch_size_type = typename batch_type::size_type;

        using container = std::vector<batch_type>;
        using const_iterator = typename container::const_iterator;
        using size_type = typename container::size_type;

        DatasetBatchManager(const DatasetManager<D> &manager, batch_size_type B)
            : batchSize_(B)
        {
            // choise indecies randomly
            std::vector<batch_size_type> choiseIndecies(manager.size());
            for (auto i = 0; i < manager.size(); i++)
            {
                choiseIndecies[i] = i;
            }
            std::random_shuffle(choiseIndecies.begin(), choiseIndecies.end());

            // copy data
            for (auto i = 0; i < manager.size(); i+=B) {
                batch_type batch;
                for (auto j = i; j < i+B; j++) {
                    batch.push_back(manager[j]);
                }
                batchList_.push_back(batch);
            }
        }

        const_iterator begin() const {
            return batchList_.begin();
        }

        const_iterator end() const {
            return batchList_.end();
        }

        size_type size() const {
            return batchList_.size();
        }

    private:
        container batchList_;
        int batchSize_;
};


#endif