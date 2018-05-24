#include "matrix.hpp"
#include "layer.hpp"
#include "neural_net.hpp"

#include <cstdio>
#include <random>
#include <functional>
#include <cmath>


void generate(int n, int m, std::vector<matrix<double>> &x, std::vector<int> &y) {

}

const int epoch = 0;

int main() {
    int n, m; scanf("%d %d", &n, &m); // number of samples, number of variables
    std::vector<matrix<double>> x;
    std::vector<int> y;
    generate(n, m, x, y);

    size_t n_layer; scanf("%zu", &n_layer);
    std::vector<size_t> nodes(n_layer);
    for (size_t i = 0; i < n_layer; ++i) scanf("%zu", &nodes[i]);
    std::vector<std::function<double(double)>> fs(n_layer - 1);
    fill(fs.begin(), fs.end(), [](const double &z) { return 1.0 / (1 + exp(-z)); });

    neural_net<double> nn(nodes, fs);

    for (int iter = 0; iter < epoch; ++iter) {
        std::vector<int> p = nn.predict(x);
        int diff = 0;
        for (size_t i = 0; i < p.size(); ++i) diff += p[i] != y[i];
        double j = nn.cost(x, y);
        fprintf(stderr, "iter = %d diff = %d cost = %.5lf\n", iter, diff, j);
        nn.backprop(x, y);
    }
    return 0;
}
