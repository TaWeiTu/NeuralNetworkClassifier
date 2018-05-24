#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <cassert>
#include <cstdio>
#include <functional>


template <typename T>
struct matrix {
    std::vector<std::vector<T>> data;
    int n, m;

    matrix();
    matrix(int, int);

    matrix<T> operator+(const matrix<T> &) const;
    matrix<T> operator-(const matrix<T> &) const;
    matrix<T> operator*(const matrix<T> &) const;
    matrix<T> operator*(const T &) const;
    matrix<T> operator^(const matrix<T> &) const;

    matrix<T> &operator+=(const matrix<T> &);
    matrix<T> &operator-=(const matrix<T> &);
    matrix<T> &operator*=(const matrix<T> &);
    matrix<T> &operator*=(const T &);
    matrix<T> &operator^=(const matrix<T> &);

    std::vector<T> &operator[](const int&);
    void resize(int, int);
    matrix<T> apply(std::function<T(T)>) const;
    matrix<T> transpose() const;

};

template <typename T> matrix<T> concate_h(matrix<T>, matrix<T>);
template <typename T> matrix<T> concate_v(matrix<T>, matrix<T>);
template <typename T> matrix<T> ones(int, int);
template <typename T> matrix<T> zeros(int, int);
template <typename T> matrix<T> identity(int);


template <typename T>
matrix<T>::matrix() {}

template <typename T>
matrix<T>::matrix(int n, int m): n(n), m(m) {
    data.resize(n);
    for (int i = 0; i < m; ++i) 
        data[i].resize(m);
} 

template <typename T>
std::vector<T>& matrix<T>::operator[](const int &i) {
    return data[i];
}

template <typename T>
matrix<T> matrix<T>::operator+(const matrix<T> &rhs) const {
    assert(n == rhs.n && m == rhs.m);
    matrix res(n, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        res.data[i][j] = data[i][j] + rhs.data[i][j];
    return res;
}

template <typename T>
matrix<T>& matrix<T>::operator+=(const matrix<T> &rhs) {
    assert(n == rhs.n && m == rhs.m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        data[i][j] += rhs.data[i][j];
    return *this;
}

template <typename T>
matrix<T> matrix<T>::operator-(const matrix<T> &rhs) const {
    assert(n == rhs.n && m == rhs.m);
    matrix<T> res(n, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        res.data[i][j] = data[i][j] - rhs.data[i][j];
    return res;
}

template <typename T>
matrix<T>& matrix<T>::operator-=(const matrix<T> &rhs) {
    assert(n == rhs.n && m == rhs.m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        data[i][j] -= rhs.data[i][j];
    return *this;
}

template <typename T>
matrix<T> matrix<T>::operator*(const T &scalar) const {
    matrix<T> res(n, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        res.data[i][j] = data[i][j] * scalar;
    return res;
}

template <typename T>
matrix<T>& matrix<T>::operator*=(const T &scalar) {
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        data[i][j] *= scalar;
    return *this;
}

template <typename T>
matrix<T> matrix<T>::operator*(const matrix<T> &rhs) const {
    assert(m == rhs.n);
    matrix<T> res(n, rhs.m);
    for (int i = 0; i < n; ++i) for (int k = 0; k < m; ++k) {
        for (int j = 0; j < rhs.m; ++j) 
            res.data[i][j] += data[i][k] * rhs.data[k][j];
    }
    return res;
}

template <typename T>
matrix<T>& matrix<T>::operator*=(const matrix<T> &rhs) {
    matrix<T> res = (*this) * rhs;
    (*this) = res;
    return *this;
}

template <typename T>
matrix<T> matrix<T>::operator^(const matrix<T> &rhs) const {
    assert(n == rhs.n && m == rhs.m);
    matrix res(n, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j)
        res.data[i][j] = data[i][j] * rhs.data[i][j];
    return res;
}

template <typename T>
matrix<T>& matrix<T>::operator^=(const matrix<T> &rhs) {
    assert(n == rhs.n && m == rhs.m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j)
        data[i][j] *= rhs.data[i][j];
    return *this;
}

template <typename T>
matrix<T> matrix<T>::apply(std::function<T(T)> f) const {
    matrix res(n, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        res.data[i][j] = f(data[i][j]);
    return res;
}

template <typename T>
void matrix<T>::resize(int new_n, int new_m) {
    n = new_n, m = new_m;
    data.resize(n);
    for (int i = 0; i < n; ++i) 
        data[i].resize(m);
}

template <typename T>
matrix<T> matrix<T>::transpose() const {
    matrix<T> res(m, n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        res.data[j][i] = data[i][j];
    return res;
}

template <typename T>
matrix<T> concate_h(matrix<T> a, matrix<T> b) {
    assert(a.n == b.n);
    matrix<T> res(a.n, a.m + b.m);
    for (int i = 0; i < a.n; ++i) {
        for (int j = 0; j < a.m; ++j) 
            res[i][j] = a[i][j];
        for (int j = 0; j < b.m; ++j) 
            res[i][j + a.m] = b[i][j];
    }
    return res;
}

template <typename T>
matrix<T> concate_v(matrix<T> a, matrix<T> b) {
    assert(a.m == b.m);
    matrix<T> res(a.n + b.n, a.m);
    for (int j = 0; j < a.m; ++j) {
        for (int i = 0; i < a.n; ++i) 
            res[i][j] = a[i][j];
        for (int i = 0; i < b.n; ++i) 
            res[i + a.n][j] = b[i][j];
    }
    return res;
}

template <typename T>
matrix<T> ones(int n, int m) {
    matrix<T> res(n, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        res[i][j] = 1;
    return res;
}

template <typename T>
matrix<T> zeros(int n, int m) {
    matrix<T> res(n, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) 
        res[i][j] = 0;
    return res;
}

template <typename T>
matrix<T> identity(int n) {
    matrix<T> res(n, n);
    for (int i = 0; i < n; ++i) 
        res[i][i] = 1;
    return res;
}

#endif
