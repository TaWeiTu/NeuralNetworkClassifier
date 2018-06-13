#ifndef TRAIN_HPP_INCLUDED
#define TRAIN_HPP_INCLUDED

#include <vector>
#include <algorithm>


template <typename T> class batch {
private:
    std::vector<std::vector<T>> _x;
    std::vector<int> _y;
    size_t _batch_size, _prv;

public:
    batch(size_t batch_size, const std::vector<std::vector<T>> &x, const std::vector<int> &y):
          _x(x), _y(y), _batch_size(batch_size), _prv(0) {}
    
    bool next(std::vector<std::vector<T>> &x, std::vector<int> &y) {
        if (_prv == _x.size()) return false;
        size_t size = std::min(_batch_size, _x.size() - _prv);
        std::vector<std::vector<T>> bx(_x.begin() + _prv, _x.begin() + _prv + size);
        std::vector<int> by(_y.begin() + _prv, _y.begin() + _prv + size);
        _prv += size;
        x = bx, y = by;
        return true;
    }

    void reload() { _prv = 0; }
};


#endif 
