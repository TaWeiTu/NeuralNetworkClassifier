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

template <typename T> class adam: public _optimizer {
private:
    double _beta1, _beta2;
    double _biased_beta1, _biased_beta2;
    std::vector<matlib::matrix<T>> _vdw, _vdb, _sdw, _sdb;

public:
    adam() {}
    adam(const std::vector<double> &param,
            const std::vector<matlib::matrix<T>> &w, const std::vector<matlib::matrix<T>> &b) {
        if (param.size() != 3u) throw std::invalid_argument("adam::adam(param should be [alpha, beta1, beta2])");
        _alpha = param[0];
        _beta1 = param[1];
        _beta2 = param[2];
        _biased_beta1 = _biased_beta2 = 1.;

        _vdw.resize(w.size()), _vdb.resize(b.size());
        _sdw.resize(w.size()), _sdb.resize(b.size());
        for (size_t i = 0; i < w.size(); ++i) {
            _vdw[i] = matlib::zeros<T>(w[i].row(), w[i].col());
            _vdb[i] = matlib::zeros<T>(b[i].row(), b[i].col());
            _sdw[i] = matlib::zeros<T>(w[i].row(), w[i].col());
            _sdb[i] = matlib::zeros<T>(b[i].row(), b[i].col());
        }
    }

    void update(std::vector<matlib::matrix<T>> &w, const std::vector<matlib::matrix<T>> &dw,
                std::vector<matlib::matrix<T>> &b, const std::vector<matlib::matrix<T>> &db) {
        _biased_beta1 *= _beta1;
        _biased_beta2 *= _beta2;

        for (size_t i = 1; i < w.size(); ++i) {
            _vdw[i] = _beta1 * _vdw[i] + (1. - _beta1) * dw[i];          
            _vdb[i] = _beta1 * _vdb[i] + (1. - _beta1) * db[i];          
            _sdw[i] = _beta2 * _sdw[i] + (1. - _beta2) * (dw[i] ^ dw[i]);          
            _sdb[i] = _beta2 * _sdb[i] + (1. - _beta2) * (db[i] ^ db[i]);          

            w[i] -= _alpha * ((_vdw[i] / (1. - _biased_beta1)) % matlib::xsqrt((_sdw[i] / (1. - _biased_beta2)) + 1e-8));
            b[i] -= _alpha * ((_vdb[i] / (1. - _biased_beta1)) % matlib::xsqrt((_sdb[i] / (1. - _biased_beta2)) + 1e-8));
        }
    }
};

}

template <class C>
using is_optimizer = typename std::is_base_of<optimizer::_optimizer, C>;


#endif
