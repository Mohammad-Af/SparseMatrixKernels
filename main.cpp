#include <iostream>
#include <fstream>
#include <algorithm>
#include <queue>
#include "csc_formatter.h"
#include <omp.h>
#include <cfloat>

using namespace std::chrono;

// naive Dense solver
vector<double> lSolveDense(CSC<double> &A, CSC<double> &b) {
    auto x = toDense(b);
    for (int j = 0; j < A.N; j++) {
        x[j] /= A.Lx[A.Lp[j]];
        for (int p = A.Lp[j] + 1; p < A.Lp[j + 1]; p++) {
            x[A.Li[p]] -= A.Lx[p] * x[j];
        }
    }
    return x;
}

void topologicalSortUtil(int v, bool visited[], stack<int> &Stack, CSC<double> &A, CSC<double> &b) {
    // Mark the current node as visited.
    visited[v] = true;

    // Recur for all the vertices adjacent to this vertex
    for (int p = A.Lp[v + 1] - 1; p > A.Lp[v]; p--)
        if (!visited[A.Li[p]])
            topologicalSortUtil(A.Li[p], visited, Stack, A, b);

    // Push current vertex to stack which stores result
    Stack.push(v);
}

// The function to do Topological Sort. It uses recursive topologicalSortUtil()
stack<int> topologicalSort(CSC<double> &A, CSC<double> &b) {
    stack<int> Stack;
    // Mark all the vertices as not visited
    bool *visited = new bool[A.N];
    for (int i = 0; i < A.N; i++)
        visited[i] = false;

    // Call the recursive helper function to store Topological Sort
    // starting from all vertices one by one
    for (const auto &i: b.Li)
        if (!visited[i])
            topologicalSortUtil(i, visited, Stack, A, b);

    return Stack;
}

vector<double> lSolveSparse(CSC<double> &A, CSC<double> &b) {
    // computes the set Reach(B) using topological sort
    stack<int> reach = topologicalSort(A, b);
    auto x = toDense(b);

    while (!reach.empty()) {
        int j = reach.top();
        reach.pop();
        x[j] /= A.Lx[A.Lp[j]];

        #pragma omp for schedule(dynamic)
        for (int p = A.Lp[j] + 1; p < A.Lp[j + 1]; p++) {
            x[A.Li[p]] -= A.Lx[p] * x[j];
        }
    }
    return x;
}

vector<double> lSolve(CSC<double> &A, CSC<double> &b, bool SPARSE) {
    if (SPARSE)
        return lSolveSparse(A, b);
    else
        return lSolveDense(A, b);
}

void measureRelativeError(vector<double> &xD, vector<double> &xS) {
    double max = -DBL_MAX;
    for (int i = 0; i < xD.size(); i++)
        max = std::max(abs(xD[i] - xS[i]) / std::max(abs(xD[i]), abs(xS[i])), max);
    cout << max;
}


void measureError(vector<double> &xD, vector<double> &xS) {
    double max = -DBL_MAX;
    for (int i = 0; i < xD.size(); i++)
        max = std::max(abs(xD[i] - xS[i]), max);
    cout << max;
}

int main(int argc, char **argv) {
    string A_PATH = "TSOPF_RS_b678_c2/TSOPF_RS_b678_c2.mtx";
    string b_PATH = "b_for_TSOPF_RS_b678_c2_b.mtx";

    // You can pass the A_PATH, b_PATH as arguments
    if (argc == 2) {
        A_PATH = argv[0];
        b_PATH = argv[1];
    }

    bool SPARSE = true; // used for sparse or dense solver
    auto A = assemble_csc_matrix(A_PATH);
    auto b = assemble_csc_matrix(b_PATH);

    auto xS = lSolve(A, b, SPARSE);
    auto xD = lSolve(A, b, !SPARSE);

//    measureError(xS, xD);

}