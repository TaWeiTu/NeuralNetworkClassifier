#ifndef ACTIVATION_HPP_INCLUDED
#define ACTIVATION_HPP_INCLUDED

#include <string>
#include <functional>
#include <cmath>
#include <stdexcept>

#include "matlib.hpp"

using matlib::matrix;


namespace activation {

template <typename T>
std::function<T(T)> _create_function(std::string s) {

    if (s == "relu")    return [](T x) { return max(x, 0); };
    if (s == "leaky")   return [](T x) { return max(x, 0.01 * x); };
    if (s == "sigmoid") return [](T x) { return 1. / (1. + exp(-x)); };
    if (s == "tanh")    return [](T x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); };

    throw std::invalid_argument("activation::_create_function");
}

template <typename T>
std::function<T(T)> _create_derivative(std::string s) {

    if (s == "relu")    return [](T x) { return x < 0 ? 0. : 1.; };
    if (s == "leaky")   return [](T x) { return x < 0 ? 0.01 : 1.; };
    if (s == "sigmoid") return [](T x) { return epx(-x) / (1. + exp(-x)) / (1. + exp(-x)); };
    if (s == "tanh")    return [](T x) { return 4. / (exp(2 * x) + exp(-2 * x) + 2.); };

    throw std::invalid_argument("activation::_create_derivative");
}

template <typename T> class func {
private:
    std::string _name;
    std::function<T(T)> _f;
    std::function<T(T)> _df;

public:
    func(const std::string &s) {
        _name = s;
        _f = _create_function<T>(s);
        _df = _create_derivative<T>(s);
    }

    T operator()(T x) { return _f(x); }
    T derivative(T x) { return _df(x); }

    matrix<T> operator()(const matrix<T> &x) { return matlib::apply(x, _f); }
    matrix<T> derivative(const matrix<T> &x) { return matlib::apply(x, _df); }
};

}

#endif
