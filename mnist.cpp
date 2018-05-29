#include "neural_net.hpp"
#include "preprocess.hpp"

#include <cstdio>
#include <vector>
#include <cmath>
#include <functional>

const int trains = 60000;
const int tests = 10000;
const int train_size = 500;
const int test_size = 500;
const int dimension = 784;

const int epoch = 50;
const double alpha = 0.01;

int main(int argc, const char **argv) {
    FILE *train_images = fopen("mnist_train_images.txt", "r"), *train_labels = fopen("mnist_train_labels.txt", "r");

    std::vector<matrix<double>> x_train(trains);
    std::vector<int> y_train(trains);

    for (int i = 0; i < trains; ++i) {
        x_train[i].resize(dimension, 1);
        for (int j = 0; j < dimension; ++j) fscanf(train_images, "%lf", &x_train[i][j][0]);
        fscanf(train_labels, "%d", &y_train[i]);
    }

    FILE *test_images = fopen("mnist_test_images.txt", "r"), *test_labels = fopen("mnist_test_labels.txt", "r");

    std::vector<matrix<double>> x_test(tests);
    std::vector<int> y_test(tests);

    for (int i = 0; i < tests; ++i) {
        x_test[i].resize(dimension, 1);
        for (int j = 0; j < dimension; ++j) fscanf(test_images, "%lf", &x_test[i][j][0]);
        fscanf(test_labels, "%d", &y_test[i]);
    }

    fclose(train_images), fclose(train_labels), fclose(test_images), fclose(test_labels);

    puts("done reading files");

    size_t n_layer; scanf("%zu", &n_layer);
    std::vector<size_t> nodes(n_layer);
    for (size_t i = 0; i < n_layer; ++i) scanf("%zu", &nodes[i]);

    puts("done reading configuration");

    std::function<double(double)> sigmoid = [](double z) { return 1. / (1 + exp(-z)); };

    std::vector<std::function<double(double)>> fs(n_layer);
    fill(fs.begin(), fs.end(), sigmoid);

    neural_net<double> nn(nodes, fs, alpha);

    using preprocess::random_sampling;
    tie(x_train, y_train) = random_sampling(x_train, y_train, train_size);
    tie(x_test, y_test) = random_sampling(x_test, y_test, test_size);

    puts("start training");
    
    for (int iter = 0; iter < epoch; ++iter) {
        printf("iter = %d cost = %.5lf\n", iter, nn.cost(x_train, y_train));
        nn.backprop(x_train, y_train);
    }

    puts("done training");

    int acc = 0;
    std::vector<int> prd = nn.predict(x_test);
    for (size_t i = 0; i < prd.size(); ++i) if (prd[i] == y_test[i]) ++acc;

    printf("accuracy = %.5lf\n", 1. * acc / prd.size());

    return 0;
}
