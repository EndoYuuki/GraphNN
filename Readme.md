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
`build`フォルダに移動した後，次のコマンドを実行する．
``` bash
./GraphNN iteration batch_size
```
`results`フォルダ下に，それぞれのOptimizerによる実行結果と，
Adamで最適化したパラメータに対しtestデータを適用した推定結果が格納される．

実行には私の環境で30分ほどかかった．

googletestを実行するためには次のコマンドを実行する．
``` bash
make test
```