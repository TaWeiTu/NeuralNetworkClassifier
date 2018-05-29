#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include "matrix.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <utility>

namespace preprocess {

    template <typename T>
    std::vector<matrix<T>> normalize(const std::vector<matrix<T>> &x) {
        std::vector<matrix<T>> res(x.size(), matrix<T>(x[0].dim));
        for (size_t i = 0; i < x[0].dim[0]; ++i) {
            T avg = 0;
            for (size_t j = 0; j < x.size(); ++j) avg += x[j][i][0];
            avg /= x.size();
            T stddev = 0;
            for (size_t j = 0; j < x.size(); ++j) stddev += (x[j][i][0] - avg) * (x[j][i][0] - avg);
            stddev /= (x.size() - 1);
            stddev = sqrt(stddev);
            for (size_t j = 0; j < x.size(); ++j) res[j][i][0] = (x[j][i][0] - avg) / stddev;
        }
        return res;
    }

    std::vector<int> _random_permutation(size_t size) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::vector<int> permutation(size);
        iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), std::mt19937(seed));
        return permutation;
    }
    
    template <typename U, typename V>
    std::pair<std::vector<U>, std::vector<V>> random_sampling(const std::vector<U> &x, const std::vector<V> &y, size_t size) {
        std::vector<int> p = _random_permutation(size);
        std::vector<U> sx(size);
        std::vector<V> sy(size);
        for (size_t i = 0; i < size; ++i) sx[i] = x[p[i]], sy[i] = y[p[i]];
        return make_pair(sx, sy);
    }

    template <typename T>
    void cross_validation(const std::vector<matrix<T>> &x, const std::vector<int> &y, double r,
                          std::vector<matrix<T>> &x_train, std::vector<int> &y_train,
                          std::vector<matrix<T>> &x_test, std::vector<int> &y_test) {
        size_t train_size = (size_t)(r * x.size());
        size_t test_size = x.size() - train_size;
        x_train.resize(train_size), y_train.resize(train_size);
        x_test.resize(test_size), y_test.resize(test_size);
        
        std::vector<int> p = _random_permutation(x.size());

        for (size_t i = 0; i < train_size; ++i) {
            x_train[i] = x[p[i]];
            y_train[i] = y[p[i]];
        }
        for (size_t i = 0; i < test_size; ++i) {
            x_test[i] = x[p[i + train_size]];
            y_test[i] = y[p[i + train_size]];
        }
    }
}

#endif