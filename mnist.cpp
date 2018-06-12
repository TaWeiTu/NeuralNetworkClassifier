#include "preprocess.hpp"
#include "network.hpp"

#include <cstdio>
#include <vector>
#include <cmath>
#include <functional>

const int trains = 60000;
const int tests = 10000;
const int train_size = 3000;
const int test_size = 500;
const int dimension = 784;

const int epoch = 100;
const double alpha = 0.1;

int main(int argc, const char **argv) {
    if (argc == 1) throw std::invalid_argument("main(configuration file is required)");
    
    FILE *train_images = fopen("mnist_train_images.txt", "r"), *train_labels = fopen("mnist_train_labels.txt", "r");
    std::vector<std::vector<long double>> x_train(trains);
    std::vector<int> y_train(trains);

    for (int i = 0; i < trains; ++i) {
        x_train[i].resize(dimension, 1);
        for (int j = 0; j < dimension; ++j) fscanf(train_images, "%Lf", &x_train[i][j]);
        fscanf(train_labels, "%d", &y_train[i]);
    }

    FILE *test_images = fopen("mnist_test_images.txt", "r"), *test_labels = fopen("mnist_test_labels.txt", "r");

    std::vector<std::vector<long double>> x_test(tests);
    std::vector<int> y_test(tests);

    for (int i = 0; i < tests; ++i) {
        x_test[i].resize(dimension, 1);
        for (int j = 0; j < dimension; ++j) fscanf(test_images, "%Lf", &x_test[i][j]);
        fscanf(test_labels, "%d", &y_test[i]);
    }

    fclose(train_images), fclose(train_labels), fclose(test_images), fclose(test_labels);

    puts("done reading files");

    FILE *config = fopen(argv[1], "r");
    size_t n_layer; fscanf(config, "%zu", &n_layer);
    std::vector<size_t> nodes(n_layer + 1);
    for (size_t i = 0; i <= n_layer; ++i) fscanf(config, "%zu", &nodes[i]);

    std::vector<std::string> func(n_layer + 1, "relu");
    func.back() = "sigmoid";
    network<long double, dimension, 10> nn(n_layer, nodes, func, alpha);
    puts("done reading configuration");


    using preprocess::random_sampling;
    using preprocess::normalize;

    tie(x_train, y_train) = random_sampling(x_train, y_train, train_size);
    tie(x_test, y_test) = random_sampling(x_test, y_test, test_size);

    x_train = normalize(x_train);
    x_test = normalize(x_test);

    puts("start training");
    
    for (int iter = 0; iter < epoch; ++iter) {
        std::vector<int> prd = nn.predict(train_size, x_train);
        long double c = nn.cost(train_size, x_train, y_train);
        int acc = 0;
        for (size_t i = 0; i < prd.size(); ++i) if (prd[i] == y_train[i]) ++acc;
        printf("iter = %d cost = %.5Lf accuracy = %.5lf\n", iter, c, 1. * acc / train_size);
        nn.fit(train_size, x_train, y_train);
    }

    puts("done training");

    int acc = 0;
    std::vector<int> prd = nn.predict(test_size, x_test);
    for (size_t i = 0; i < prd.size(); ++i) if (prd[i] == y_test[i]) ++acc;

    printf("accuracy = %.5lf\n", 1. * acc / prd.size());

    return 0;
}
