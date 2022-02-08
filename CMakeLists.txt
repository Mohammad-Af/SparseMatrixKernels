cmake_minimum_required(VERSION 3.21)
project(SparseMatrixKernels)

include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
set(CMAKE_CXX_STANDARD 14)

add_executable(SparseMatrixKernels main.cpp csc_formatter.h)
