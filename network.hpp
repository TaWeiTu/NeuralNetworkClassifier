#ifndef NETWORK_HPP_INCLUDED
#define NETWORK_HPP_INCLUDED

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cmath>
#include <type_traits>

#include "matlib.hpp"
#include "activation.hpp"
#include "optimizer.hpp"

using activation::func;

template <typename T, size_t F, size_t C, 
          class Opt, typename = std::enable_if_t<is_optimizer<Opt>::value>> class network {
private:
    size_t _layer;
    std::vector<size_t> _nodes;
    std::vector<func<T>> _g;
    std::vector<matlib::matrix<T>> _w, _z, _a, _b;
    std::vector<matlib::matrix<T>> _dw, _dz, _da, _db;
    Opt _optimizer;

    void _forward(const matlib::matrix<T> &x) {
        _a[0] = x;
        for (size_t i = 1; i <= _layer; ++i) {
            _z[i] = _w[i] * _a[i - 1];
            _z[i] += _b[i].expand(_z[i]);
            _a[i] = _g[i](_z[i]);
        }
    }
    void _backward(int m, const matlib::matrix<T> &x, const matlib::matrix<T> &y) {
        _forward(x);
        _da[_layer] = -(y / _a[_layer]) + ((1. - y) / (1. - _a[_layer]));
        for (size_t i = _layer; i >= 1; --i) {
            _dz[i] = _da[i] ^ _g[i].derivative(_z[i]);
            _dw[i] = (_dz[i] * _a[i - 1].transpose()) / m;
            _db[i] = matlib::sum(_dz[i], 1) / m;
            _da[i - 1] = _w[i].transpose() * _dz[i];
        }
        _optimizer.update(_w, _dw, _b, _db);
    }

    matlib::matrix<T> _convert(const std::vector<std::vector<T>> &x) const {
        matlib::matrix<T> res(x[0].size(), x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t j = 0; j < x[0].size(); ++j) res(j, i) = x[i][j];
        }
        return res;
    }

    std::vector<int> _resolve() const {
        std::vector<int> res(_a[_layer].col());
        for (size_t c = 0; c < _a[_layer].col(); ++c) {
            int p = -1;
            for (size_t i = 0; i < _a[_layer].row(); ++i) {
                if (p == -1 || _a[_layer](i, c) > _a[_layer](p, c)) p = i;
            }
            res[c] = p;
        }
        return res;
    }

    matlib::matrix<T> _construct(const std::vector<int> &y) const {
        matlib::matrix<T> res(C, y.size());
        for (size_t i = 0; i < y.size(); ++i) res(y[i], i) = 1.;
        return res;
    }

public:
    network() {
        _layer = 0;
        _nodes.clear();
        _g.clear();
        _w.clear(), _z.clear(), _a.clear(), _b.clear();
        _dw.clear(), _dz.clear(), _db.clear();
    }
    network(size_t layer, const std::vector<size_t> nodes, const std::vector<std::string> fname, const std::vector<double> &param): 
        _layer(layer), _nodes(nodes) {

        if (nodes.size() != layer + 1) throw std::length_error("network::network(nodes.size() != layer + 1)");
        if (fname.size() != layer + 1) throw std::length_error("network::network(fname,size() != layer + 1)");

        if (nodes.back() != C)  throw std::invalid_argument("network::network(number of output units must be equal to number of classes)");
        if (nodes.front() != F) throw std::invalid_argument("network::network(number of input untis must be equal to number of features)");

        _g.resize(layer + 1);
        _w.resize(layer + 1), _z.resize(layer + 1), _a.resize(layer + 1), _b.resize(layer + 1);
        _dw.resize(layer + 1), _dz.resize(layer + 1), _da.resize(layer + 1), _db.resize(layer + 1);

        for (size_t i = 1; i <= _layer; ++i) {
            _g[i] = func<T>(fname[i]);

            _w[i] = matlib::randomized<T>(nodes[i], nodes[i - 1], sqrt(1. / nodes[i - 1]));
            _b[i] = matlib::randomized<T>(nodes[i], 1, sqrt(1. / nodes[i - 1]));
            _z[i].reshape(nodes[i]);
            _a[i].reshape(nodes[i]);

            _dw[i].reshape(nodes[i], nodes[i - 1]);
            _dz[i].reshape(nodes[i]);
            _da[i].reshape(nodes[i]);
            _db[i].reshape(nodes[i]);
        }
        _optimizer = Opt(param, _w, _b);
    }

    void set_alpha(double alpha) { _optimizer.set_alpha(alpha); }

    std::vector<int> predict(int m, const std::vector<std::vector<T>> &x) {
        if (x[0].size() != F) throw std::invalid_argument("network::fit(input size must to equal to number of features)");

        matlib::matrix<T> input = _convert(x);
        _forward(input);
        return _resolve();
    }

    T cost(int m, const std::vector<std::vector<T>> &x, const std::vector<int> &y) {
        if ((size_t)*max_element(y.begin(), y.end()) >= C) throw std::invalid_argument("network::fit(output should ranges in [0, C))");
        if (x[0].size() != F) throw std::invalid_argument("network::fit(input size must to equal to number of features)");

        matlib::matrix<T> input = _convert(x);
        matlib::matrix<T> output = _construct(y);
        _forward(input);

        T res = 0.;
        for (size_t i = 0; i < output.row(); ++i) {
            for (size_t j = 0; j < output.col(); ++j) {
                if (fabs(output(i, j)) > 1e-9) res -= log(std::max((T)0.001, _a[_layer](i, j))) / m;
                else res -= log(std::max((T)0.001, 1. - _a[_layer](i, j))) / m;
            }
        }
        return res;
    }

    void fit(int m, const std::vector<std::vector<T>> &x, const std::vector<int> &y) {
        if ((size_t)*max_element(y.begin(), y.end()) >= C) throw std::invalid_argument("network::fit(output should ranges in [0, C))");
        if (x[0].size() != F) throw std::invalid_argument("network::fit(input size must to equal to number of features)");

        matlib::matrix<T> input = _convert(x);
        matlib::matrix<T> output = _construct(y);
        _backward(m, input, output);
    }
};

#endif
