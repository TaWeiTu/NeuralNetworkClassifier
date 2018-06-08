#ifndef MATLIB_HPP_INCLUDED
#define MATLIB_HPP_INCLUDED

#include <vector>
#include <functional>
#include <stdexcept>
#include <random>


namespace matlib {

template <typename T> class matrix {
private:
    std::vector<T> _data;
    size_t _r, _c;

public:
    matrix(): _r(0), _c(0) { _data.clear(); }
    matrix(size_t r, size_t c = 1): _r(r), _c(c) { _data.assign(r * c, 0); }
    matrix(size_t r, size_t c, T val): _r(r), _c(c) { _data.assign(r * c, val); }
    matrix(const matrix &rhs): _data(rhs.data), _r(rhs._r), _c(rhs._c) {}

    matrix(size_t r, size_t c, std::function<T(size_t, size_t)> f): _r(r), _c(c) {
        _data.assign(r * c);
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) _data[i][j] = f(i, j);
        }
    }

    void reshape(size_t r, size_t c = 1) {
        _data.assign(r * c, 0);
        _r = r, _c = c;
    }
    void reshape(size_t r, size_t c, T val) {
        _data.assign(r * c, val);
        _r = r, _c = c;
    }

    T operator()(size_t i, size_t j = 0) const { 
        if (i >= _r || j >= _c) throw std::out_of_range("matrix::operator()");
        return _data[i * _c + j]; 
    }
    T& operator()(size_t i, size_t j = 0) { 
        if (i >= _r || j >= _c) throw std::out_of_range("matrix::operator()");
        return _data[i * _c + j]; 
    }

    size_t row() const { return _r; }
    size_t col() const { return _c; }

    matrix<T> transpose() const {
        matrix<T> res(_c, _r);
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) res(j, i) = (*this)(i, j);
        }
        return res;
    }

    template <typename U> 
    matrix<T> operator+(const matrix<U> &rhs) const {
        if (_r != rhs.row() || _c != rhs.col()) throw std::length_error("matrix::operator+");
        matrix<T> res(_r, _c);
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) res(i, j) = (*this)(i, j) + rhs(i, j);
        }
        return res;
    }
    template <typename U> 
    matrix<T> operator-(const matrix<U> &rhs) const {
        if (_r != rhs.row() || _c != rhs.col()) throw std::length_error("matrix::operator-");
        matrix<T> res(_r, _c);
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) res(i, j) = (*this)(i, j) - rhs(i, j);
        }
        return res;
    }
    template <typename U>
    matrix<T> operator*(const matrix<U> &rhs) const {
        if (_c != rhs.row()) throw std::length_error("matrix::operator*");
        matrix<T> res(_r, rhs.col());
        for (size_t i = 0; i < _r; ++i) {
            for (size_t k = 0; k < _c; ++k) {
                for (size_t j = 0; j < rhs.col(); ++j) res(i, j) += (*this)(i, k) * rhs(k, j);
            }
        }
        return res;
    }
    template <typename U>
    matrix<T> operator*(U c) const {
        matrix<T> res(_r, _c);
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) res(i, j) = (*this)(i, j) * c;
        }
        return res;
    }
    template <typename U>
    matrix<T> operator/(U c) const {
        matrix<T> res(_r, _c);
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) res(i, j) = (*this)(i, j) / c;
        }
        return res;
    }
    template <typename U>
    matrix<T> operator^(const matrix<U> &rhs) const {
        if (_r != rhs.row() || _c != rhs.col()) throw std::length_error("matrix::operator^");
        matrix<T> res(_r, _c);
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) res[i][j] = (*this)(i, j) * rhs(i, j);
        }
        return res;
    }

    template <typename U>
    matrix<T> &operator+=(const matrix<U> &rhs) {
        if (_r != rhs.row() || _c != rhs.col()) throw std::length_error("matrix::operator+=");
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) (*this)(i, j) += rhs(i, j);
        }
        return (*this);
    }
    template <typename U>
    matrix<T> &operator-=(const matrix<U> &rhs) {
        if (_r != rhs.row() || _c != rhs.col()) throw std::length_error("matrix::operator-=");
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) (*this)(i, j) -= rhs(i, j);
        }
        return (*this);
    }
    template <typename U>
    matrix<T> &operator*=(const matrix<U> &rhs) {
        if (_c != rhs.row()) throw std::length_error("matrix::operator*=");
        matrix<T> res = (*this) * rhs;
        (*this) = res;
        return (*this);
    }
    template <typename U>
    matrix<T> &operator*=(U c) {
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) (*this)(i, j) *= c;
        }
        return (*this);
    }
    template <typename U>
    matrix<T> &operator/=(U c) const {
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) (*this)(i, j) /= c;
        }
        return (*this);
    }
    template <typename U>
    matrix<T> &operator^=(const matrix<U> &rhs) {
        if (_r != rhs.row() || _c != rhs.col()) throw std::length_error("matrix::operator^=");
        for (size_t i = 0; i < _r; ++i) {
            for (size_t j = 0; j < _c; ++j) (*this)(i, j) *= rhs(i, j);
        }
        return (*this);
    }

    template <typename U>
    matrix<T> expand(const matrix<U> &rhs) const {
        if (_r != rhs.row() || _c > rhs.col() || rhs.col() % _c != 0) throw std::length_error("matrix::expand");
        matrix<T> res(rhs.row(), rhs.col());
        for (size_t c = 0; c < rhs.col(); c += _c) {
            for (size_t i = 0; i < _r; ++i) {
                for (size_t j = 0; j < _c; ++j) res(i, j + c) = (*this)(i, j);
            }
        }
        return res;
    }

};

template <typename T> matrix<T> zeros(size_t r, size_t c = 1) { return matrix<T>(r, c, 0); }
template <typename T> matrix<T> ones(size_t r, size_t c = 1) { return matrix<T>(r, c, 1); }
template <typename T> matrix<T> eye(size_t s) { return matrix<T>(s, s, [](size_t r, size_t c) -> T { return r == c; }); }

template <typename T, typename U>
matrix<U> operator*(T c, const matrix<U> &u) { return u * c; }

template <typename T>
matrix<T> apply(const matrix<T> &x, std::function<T(T)>) {
    matrix<T> res(x);
    for (size_t i = 0; i < x.row(); ++i) {
        for (size_t j = 0; j < x.col(); ++j) res(i, j) = f(x(i, j));
    }
    return res;
}

template <typename T>
T sum(const matrix<T> &x) {
    T res = 0.;
    for (size_t i = 0; i < x.row(); ++i) {
        for (size_t j = 0; j < x.col(); ++j) res += x(i, j);
    }
    return res;
}

template <typename T>
matrix<T> _sum_row(const matrix<T> &x) {
    matrix<T> res(1, x.col());
    for (size_t c = 0; c < x.col(); ++c) {
        for (size_t r = 0; r < x.row(); ++r) res(0, c) += x(r, c);
    }
    return res;
}

template <typename T>
matrix<T> _sum_col(const matrix<T> &x) {
    matrix<T> res(x.row(), 1);
    for (size_t r = 0; r < x.row(); ++r) {
        for (size_t c = 0; r < x.col(); ++c) res(r) += x(r, c);
    }
    return res;
}

template <typename T>
matrix<T> sum(const matrix<T> &x, int flag) {
    if (flag == 0) return _sum_row<T>(x);
    if (flag == 1) return _sum_col<T>(x);
    throw std::invalid_argument("matlib::sum");
}

template <typename T>
matrix<T> randomized(size_t r, size_t c = 1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1., 1.);
    matrix<T> res(r, c);
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) res(i, j) = dis(gen);
    }
    return res;
}

}
#endif
