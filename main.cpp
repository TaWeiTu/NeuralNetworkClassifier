#include "matrix.hpp"
#include "layer.hpp"
#include "neural_net.hpp"
// #include "preprocess.hpp"

#include <cstdio>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>


void generate(int n, int m, std::vector<matrix<double>> &x, std::vector<int> &y) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    std::vector<double> coef(m + 1);
    for (int i = 0; i < m + 1; ++i) coef[i] = dis(gen);
    x.resize(n); y.resize(n, 0);
    std::vector<double> fs, ff;
    for (int i = 0; i < n; ++i) {
        double f = coef[0];
        x[i].resize(m, 1);
        for (int j = 0; j < m; ++j) x[i][j][0] = dis(gen), f += coef[j + 1] * x[i][j][0];
        fs.emplace_back(f); ff.emplace_back(f);
    }
    std::sort(fs.begin(), fs.end());
    for (int i = 0; i < n; ++i) {
        if (ff[i] < fs[n / 3]) y[i] = 0;
        else if (ff[i] < fs[n * 2 / 3]) y[i] = 1;
        else y[i] = 2;
    }
    for (int i = 0; i < n; ++i) printf("%d ", y[i]); puts("");
}

const int epoch = 200;
const double alpha = 0.01;

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

    neural_net<double> nn(nodes, fs, alpha);

    for (int iter = 0; iter < epoch; ++iter) {
        std::vector<int> p = nn.predict(x);
        int diff = 0;
        for (size_t i = 0; i < p.size(); ++i) diff += p[i] != y[i];
        double j = nn.cost(x, y);
        printf("iter = %d diff = %d cost = %.5lf\n", iter, diff, j);
        nn.backprop(x, y);
    }
    std::vector<int> p = nn.predict(x);
    for (int i = 0; i < n; ++i) printf("%d ", y[i]);
    puts("");
    puts("");
    for (int i = 0; i < n; ++i) printf("%d ", p[i]);
    puts("");
    return 0;
}
