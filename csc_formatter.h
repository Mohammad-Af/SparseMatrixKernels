#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <omp.h>

using namespace std;

template <typename T> struct CSC {
    int N;
    vector<T> Lx;
    vector<int> Li;
    vector<int> Lp;
};


// creates a CSC from matrix market format
CSC<double> assemble_csc_matrix(std::string filePath) {
    int M, N, L;
    CSC<double> matrix;
    std::ifstream fin(filePath);
    // Ignore headers and comments
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters
    fin >> M >> N >> L;
    matrix.N = M;
    int last_col = 1;
    matrix.Lp.push_back(0);
    for (int l = 0; l < L; l++) {
        int row, col;
        double data;
        fin >> row >> col >> data;
        if (row < col)
            continue;
        matrix.Li.push_back(row - 1);
        matrix.Lx.push_back(data);
        if (col > last_col) {
            last_col = col;
            matrix.Lp.push_back(matrix.Li.size() - 1);
        }
    }
    matrix.Lp.push_back(matrix.Li.size());
    fin.close();
    return matrix;
}


// converts a CSC vector to Dense vector
vector<double> toDense(CSC<double> &b){
    vector<double> x(b.N, 0.0);
    for (int l = 0; l < b.Li.size(); l++)
        x[b.Li[l]] = b.Lx[l];
    return x;
}



