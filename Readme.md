# Graph Neural Network
## Environment
### My Experiment Environment
- Windows 10 Home Edition
- msys64

### Required
- GCC 8.3.0
- CMake 
- Eigen 3.3.7
- googletest 1.8.1

## Build
``` bash
mkdir build && cd build
cmake ..
make
```

## Execute
First, you need preparing the dataset consists of train, validation and test.
The training data is needed to learn your model. It consist of labels and graph information.
The validation data is needed to compare the performance for each optimizers.
The testing data is used for computing outputs you want to know.

I assume your folder tree has become following structure.
dataset
|_ train
| |_ train
| |_ valid
|_ test

Each folder has graph data and label data (but test data doesn't have label data).
You can refer an example of input format of graph data in `random_graph.txt`.
It consist of the number of vertecies and connections.

In train/valid dataset, you should name the graph and label file name with `N_graph.txt`, `N_label.txt` respectively (N is an id of data).
Each label file has one number corresponding to a graph label.

Move to `build` folder and execute next command.
``` bash
./GraphNN iteration batch_size dataset_path output_path
```
then, the results of optimization and the outputs with test data under `results` folder.

### For GoogleTest
``` bash
make test
```
or 
``` bash
ctest
```
