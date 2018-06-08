#ifndef NETWORK_HPP_INCLUDED
#define NETWORK_HPP_INCLUDED

#include <vector>
#include <stdexcept>

#include "matlib.hpp"
#include "activation.hpp"

using activation::func;
using matlib::matrix;


template <typename T> 
class network {
private:
    size_t layer;
    std::vector<size_t> nodes;
    std::vector<func<T>> g;
    std::vector<matrix<T>> w, z, a, b;
    std::vector<matrix<T>> dw, dz, da, db;

    void _forward(const matrix<T> &x) {
        a[0] = x;
        for (int i = 1; i <= layer; ++i) {
            z[i] = w[i] * a[i - 1];
            z[i] += b[i].expand(z[i]);
            a[i] = g[i](z[i]);
        }
    }
    void _backward(int m, const matrix<T> &x, const matrix<T> &y) {
        _forward(x);
        // da[layer] = a[layer] - y;
        for (size_t i = layer; i >= 1; --i) {
            dz[i] = da[i] ^ g[i].derivative(z[i]);
            dw[i] = (dz[i] * a[i - 1].transpose()) / m;
            db[i] = matlib::sum(dz[i], 1) / m;
            da[i - 1] = w[i].transpose() * dz[i];
        }
    }

public:
    network() {
        layer = 0;
        nodes = 0;
        g.clear();
        w.clear(), z.clear(), a.clear(), b.clear();
        dw.clear(), dz.clear(), db.clear();
    }
    network(size_t layer, const std::vector<size_t> nodes, const std::vector<std::string> fname): layer(layer), nodes(nodes) {
        if (nodes.size() != layer + 1) throw std::length_error("network::network");
        if (fname.size() != layer + 1) throw std::length_error("network::network");

        g.resize(layer + 1);
        w.resize(layer + 1), z.resize(layer + 1), a.resize(layer + 1), b.resize(layer + 1);
        dw.resize(layer + 1), dz.resize(layer + 1), da.resize(layer + 1), db.resize(layer + 1);

        for (size_t i = 1; i <= layer; ++i) {
            g[i] = func<T>(fname[i]);

            w[i] = matlib::randomized<T>(nodes[i], nodes[i - 1]);
            b[i] = matlib::randomized<T>(nodes[i]);
            z[i].reshape(nodes[i]);
            a[i].reshape(nodes[i]);

            dw[i].reshape(nodes[i], nodes[i - 1]);
            dz[i].reshape(nodes[i]);
            da[i].reshape(nodes[i]);
            db[i].reshape(nodes[i]);
        }
    }
};

#endif
