# 🚀 YallaSQL  
GPU-powered SQL processor for CSV files 
 
✅ Lightning-fast (GPU go brrrr 💨)

✅ Works on CSV files (No fuss, just data 📊)

✅ Easy setup (Because who likes headaches? 🤯)

## 🛠️ Installation
1. 📦 Download and Install [CMake](https://cmake.org/download/)
2. 🎮 Install CUDA Toolkit
3. install duckdb
```bash
cd vendor/duckdb
make # or  "make debug"
```
4. install tools
```bash
conda install -c conda-forge quill
# todo add replexx
```

## 🏗️ How to Run ?
```bash
mkdir logs
mkdir build   && cd build  
cmake ..  -G Ninga  && ninja
./yallasql_cli  
```

## SSB-Benchmark
```bash
cd benchmark/ssb-dbgen
make
./dbgen -s 1 -T c
./dbgen -s 1 -T p
./dbgen -s 1 -T s
./dbgen -s 1 -T d
./dbgen -s 1 -T l
./python benchmark/script/ssb_to_ta_format.py
```