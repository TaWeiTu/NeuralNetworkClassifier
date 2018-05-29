#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "layer.hpp"
#include "matrix.hpp"

#include <functional>
#include <vector>
#include <cmath>


template <typename T>
struct neural_net {
    size_t inp_size, n_layer;
    std::vector<layer<T>> net;
    T alpha;

    neural_net();
    neural_net(const std::vector<size_t> &, T);
    neural_net(const std::vector<size_t> &, const std::vector<std::function<T(T)>> &, T);

    matrix<T> forward(const matrix<T> &);
    void backprop(const std::vector<matrix<T>>&, const std::vector<int> &);
    T cost(const std::vector<matrix<T>> &, const std::vector<int> &);
    std::vector<int> predict(const std::vector<matrix<T>> &);
    void update(const std::vector<matrix<T>> &);
};


template <typename T>
neural_net<T>::neural_net() {}

template <typename T>
neural_net<T>::neural_net(const std::vector<size_t> &nodes, T alpha): alpha(alpha) {
    inp_size = nodes[0];
    net.emplace_back(0, 0);
    for (size_t i = 1; i < nodes.size(); ++i) 
        net.emplace_back(nodes[i - 1], nodes[i]);
    n_layer = nodes.size();
}

template <typename T>
neural_net<T>::neural_net(const std::vector<size_t> &nodes, const std::vector<std::function<T(T)>> &f, T alpha): alpha(alpha) {
    inp_size = nodes[0];
    net.emplace_back(0, 0);
    for (size_t i = 1; i < nodes.size(); ++i) 
        net.emplace_back(nodes[i - 1], nodes[i], f[i - 1]);
    n_layer = nodes.size();
}

template <typename T>
matrix<T> neural_net<T>::forward(const matrix<T> &input) {
    matrix<T> last = input;
    net[0].a = input;
    for (size_t i = 1; i < net.size(); ++i) {
        matrix<T> output = net[i].output(last);
        net[i].a = output;
        last = output;
    }
    return last;
}

template <typename T>
void neural_net<T>::backprop(const std::vector<matrix<T>> &x, const std::vector<int> &y) {
    std::vector<matrix<T>> dlt(n_layer);
    for (size_t k = 0; k < n_layer; ++k) dlt[k] = zeros<T>(net[k].theta.dim);
    for (size_t k = 0; k < x.size(); ++k) {
        std::vector<matrix<T>> err(n_layer);
        for (size_t i = 0; i < n_layer; ++i) err[i] = zeros<T>(net[i].nodes, 1);
        matrix<T> vy(net[n_layer - 1].nodes, 1, [y, k](size_t r, size_t c) { return r == y[k]; });
        matrix<T> f = forward(x[k]);
        err[n_layer - 1] = f - vy;
        for (size_t i = n_layer - 2; i >= 1; --i) {
            err[i] = (net[i + 1].theta.transpose() * err[i + 1]).slice(1, -1, 0, -1);        
            err[i] ^= (net[i].a ^ (ones<T>(net[i].a.dim) - net[i].a));
        }
        for (size_t i = n_layer - 1; i >= 1; --i) 
            dlt[i] += err[i] * concate_v(ones<T>(1, 1), net[i - 1].a).transpose();
    }
    update(dlt);
}

template <typename T>
void neural_net<T>::update(const std::vector<matrix<T>> &dlt) {
    for (size_t i = 1; i < n_layer; ++i) 
        dlt[i].debug(),
        net[i].theta -= dlt[i] * alpha;
}

template <typename T>
T neural_net<T>::cost(const std::vector<matrix<T>> &x, const std::vector<int> &y) {
    T res = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        matrix<T> z = forward(x[i]);
        for (size_t j = 0; j < z.dim[0]; ++j) {
            if (j == y[i]) res += log(z[j][0]) / x.size();
            else res += log(1 - z[j][0]) / x.size();
        } 
    }
    return -res;
}

template <typename T>
std::vector<int> neural_net<T>::predict(const std::vector<matrix<T>> &x) {
    std::vector<int> res(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        matrix<T> f = forward(x[i]);
        res[i] = -1;
        for (size_t j = 0; j < f.dim[0]; ++j) if (res[i] == -1 || f[j][0] > f[res[i]][0]) res[i] = j;
    }
    return res;
}

#endif
