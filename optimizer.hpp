#ifndef OPTIMIZER_HPP_INCLUDED
#define OPTIMIZER_HPP_INCLUDED

#include "matlib.hpp"

#include <vector>
#include <type_traits>
#include <stdexcept>
#include <iostream>


namespace optimizer {
    
class _optimizer {
protected:
    double _alpha;

public:
    void set_alpha(double alpha) { _alpha = alpha; }
};


template <typename T> class vanilla: public _optimizer {
public:
    vanilla() {}
    vanilla(const std::vector<double> &param, 
            const std::vector<matlib::matrix<T>> &w, const std::vector<matlib::matrix<T>> &b) {
        if (param.size() != 1u) throw std::invalid_argument("vanilla::vailla(param should be [alpha])");
        _alpha = param[0];
    }

    void update(std::vector<matlib::matrix<T>> &w, const std::vector<matlib::matrix<T>> &dw,
                std::vector<matlib::matrix<T>> &b, const std::vector<matlib::matrix<T>> &db) const {
        for (size_t i = 1; i < w.size(); ++i) {
            w[i] -= _alpha * dw[i];
            b[i] -= _alpha * db[i];
        }        
    }
};

template <typename T> class momentum: public _optimizer {
private:
    double _beta;
    std::vector<matlib::matrix<T>> _vdw, _vdb;

public:
    momentum() {}
    momentum(const std::vector<double> &param,
             const std::vector<matlib::matrix<T>> &w, const std::vector<matlib::matrix<T>> &b) {
        if (param.size() != 2u) throw std::invalid_argument("momentum::momentum(param should be [alpha, beta])");
        _alpha = param[0];
        _beta = param[1];

        _vdw.resize(w.size()), _vdb.resize(b.size());
        for (size_t i = 0; i < w.size(); ++i) {
            _vdw[i] = matlib::zeros<T>(w[i].row(), w[i].col());
            _vdb[i] = matlib::zeros<T>(b[i].row(), b[i].col());
        }
    }

    void update(std::vector<matlib::matrix<T>> &w, const std::vector<matlib::matrix<T>> &dw,
                std::vector<matlib::matrix<T>> &b, const std::vector<matlib::matrix<T>> &db) {
        for (size_t i = 1; i < w.size(); ++i) {
            _vdw[i] = _beta * _vdw[i] + (1. - _beta) * dw[i];          
            _vdb[i] = _beta * _vdb[i] + (1. - _beta) * db[i];          
            w[i] -= _alpha * _vdw[i];
            b[i] -= _alpha * _vdb[i];
        }
    }
};

template <typename T> class RMSprop: public _optimizer {
private:
    double _beta;
    std::vector<matlib::matrix<T>> _sdw, _sdb;

public:
    RMSprop() {}
    RMSprop(const std::vector<double> &param,
            const std::vector<matlib::matrix<T>> &w, const std::vector<matlib::matrix<T>> &b) {
        if (param.size() != 2u) throw std::invalid_argument("RMSprop::RMSprop(param should be [alpha, beta])");
        _alpha = param[0];
        _beta = param[1];

        _sdw.resize(w.size()), _sdb.resize(b.size());
        for (size_t i = 0; i < w.size(); ++i) {
            _sdw[i] = matlib::zeros<T>(w[i].row(), w[i].col());
            _sdb[i] = matlib::zeros<T>(b[i].row(), b[i].col());
        }
    }

    void update(std::vector<matlib::matrix<T>> &w, const std::vector<matlib::matrix<T>> &dw,
                std::vector<matlib::matrix<T>> &b, const std::vector<matlib::matrix<T>> &db) {
        for (size_t i = 1; i < w.size(); ++i) {
            _sdw[i] = _beta * _sdw[i] + (1. - _beta) * (dw[i] ^ dw[i]);          
            _sdb[i] = _beta * _sdb[i] + (1. - _beta) * (db[i] ^ db[i]);          
            w[i] -= _alpha * (dw[i] % matlib::xsqrt(_sdw[i] + 1e-8));
            b[i] -= _alpha * (db[i] % matlib::xsqrt(_sdb[i] + 1e-8));
        }
    }
};

}

template <class C>
using is_optimizer = typename std::is_base_of<optimizer::_optimizer, C>;


#endif
