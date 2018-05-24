#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "layer.hpp"
#include "matrix.hpp"

#include <functional>
#include <vector>
#include <cmath>

template <typename T>
struct neural_net {
    int inp_size, n_layer;
    std::vector<layer<T>> net;
    neural_net();
    neural_net(const std::vector<int> &);
    neural_net(const std::vector<int> &, const std::vector<std::function<T(T)>> &);
    matrix<T> forward(const matrix<T> &) const;
    void backprop(const matrix<T>&, const matrix<T> &);
    T error(const matrix<T> &, const matrix<T> &);
    std::vector<int> predict(const std::vector<matrix<T>> &) const;
};


template <typename T>
neural_net<T>::neural_net() {}

template <typename T>
neural_net<T>::neural_net(const std::vector<int> &nodes) {
    inp_size = nodes[0];
    net.emplace_back(0, 0);
    for (int i = 1; i < nodes.size(); ++i) 
        net.emplace_back(nodes[i - 1], nodes[i]);
    n_layer = (int)nodes.size();
}

template <typename T>
neural_net<T>::neural_net(const std::vector<int> &nodes, const std::vector<std::function<T(T)>> &f) {
    inp_size = nodes[0];
    net.emplace_back(0, 0);
    for (int i = 1; i < nodes.size(); ++i) 
        net.emplace_back(nodes[i - 1], nodes[i], f[i - 1]);
    n_layer = (int)nodes.size();
}

template <typename T>
matrix<T> neural_net<T>::forward(const matrix<T> &input) const {
    matrix<T> last = input;
    for (int i = 0; i < net.size(); ++i) {
        matrix<T> output = net[i].output(last);
        net[i].a = output;
        last = output;
    }
    return last;
}

template <typename T>
void neural_net<T>::backprop(const matrix<T> &x, const matrix<T> &y) {
    std::vector<matrix<T>> err(n_layer);
    err[n_layer - 1] = forward(x) - y;
    for (int i = n_layer - 2; i >= 1; --i) {
        err[i] = net[i + 1].theta.transpose() * err[i + 1];        
        err[i] ^= (net[i].a ^ (ones<T>(net[i].a.n, net[i].a.m) - net[i].a));
    }
    for (int i = n_layer - 1; i >= 1; --i) {
        matrix<T> dlt = err[i] * net[i - 1].a.transpose();
        net[i].theta += dlt;
    }
}

template <typename T>
T neural_net<T>::error(const matrix<T> &x, const matrix<T> &y) {
    matrix<T> output = predict(x);
    T res = 0;
    for (int i = 0; i < y.n; ++i) res += y[i][0] * log(output[i][0]) + (1 - y[i][0]) * log(1 - output[i][0]);
    return -res;
}

#endif
