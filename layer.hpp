#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"

#include <functional>


template <typename T>
struct layer {
    std::function<T(T)> g;
    size_t nodes;
    matrix<T> theta, a;

    layer(size_t, size_t, std::function<T(T)>);
    layer(size_t, size_t);
    layer();
    matrix<T> output(const matrix<T> &) const;
};


template <typename T>
layer<T>::layer() {}

template <typename T>
layer<T>::layer(size_t prv, size_t cur, std::function<T(T)> f) {
    theta.randomized(cur, prv + 1);
    nodes = cur;
    g = f;
}

template <typename T>
layer<T>::layer(size_t prv, size_t cur) {
    theta.randomized(cur, prv + 1);
    nodes = cur;
    g = [](const T &x) { return x; };
}

template <typename T>
matrix<T> layer<T>::output(const matrix<T> &prv) const {
    matrix<T> z = theta * concate_v(ones<T>(1, 1), prv);
    matrix<T> a = z.apply(g);
    return a;
}

#endif
