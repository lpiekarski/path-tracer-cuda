#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "rtc.h"
#include "arithmetic_tuple.h"
#include "vector.h"

_RTC_BEGIN
    // ALIAS TEMPLATE matrix
template <size_t N,
    size_t M>
    using matrix = arithmetic_tuple<vector<M>, N>;

    // FUNCTION TEMPLATE identity
template <size_t N,
    size_t M>
    matrix<N, M> identity() {
        matrix<N, M> ret;
        size_t i = 0;
        while (i < N && i < M) {
            ret[i][i] = 1;
            ++i;
        }
        return ret;
    }

    // FUNCTION TEMPLATE row
template <size_t N,
    size_t M>
    vector<M> row(const matrix<N, M>& m, size_t idx) {
        return m[idx];
    }

    // FUNCTION TEMPLATE column
template <size_t N, 
    size_t M>
    vector<N> column(const matrix<N, M>& m, size_t idx) {
        vector<N> ret;
        for (size_t i = 0; i < N; ++i)
            ret[i] = m[i][idx];
        return ret;
    }

    // FUNCTION TEMPLATE operator*
template <size_t N,
    size_t M,
    size_t K>
    matrix<N, K> operator*(const matrix<N, M>& m1, const matrix<M, K>& m2) {
        matrix<N, K> ret;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < K; ++j)
                ret[i][j] = dot(row(m1, i), column(m2, j));
        return ret;
    }

    // FUNCTION TEMPLATE operator*
template <size_t N,
    size_t M>
    vector<N> operator*(const matrix<N, M>& m1, const vector<M>& v) {
        vector<N> ret;
        for (size_t i = 0; i < N; ++i)
            ret[i] = dot(row(m1, i), v);
        return v;
    }
_RTC_END

#endif /* __MATRIX_H__ */