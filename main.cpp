#include "matrix.hpp"
#include "layer.hpp"
#include "neural_net.hpp"
#include "preprocess.hpp"

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

const int epoch = 1000;
const double alpha = 0.01;
const double cross_valid = 0.7;

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

    using preprocess::normalize;
    using preprocess::cross_validation;
    x = normalize(x);

    std::vector<matrix<double>> x_train, x_test;
    std::vector<int> y_train, y_test;
    cross_validation(x, y, cross_valid, x_train, y_train, x_test, y_test);

    for (int iter = 0; iter < epoch; ++iter) {
        printf("iter = %d cost = %.10lf\n", iter, nn.cost(x_train, y_train));
        nn.backprop(x_train, y_train);
    }
    std::vector<int> prd = nn.predict(x_test);
    int acc = 0;
    for (size_t i = 0; i < x_test.size(); ++i) if (prd[i] == y_test[i]) ++acc;
    printf("accuracy = %.5lf\n", 1.0 * acc / prd.size());
    return 0;
}
