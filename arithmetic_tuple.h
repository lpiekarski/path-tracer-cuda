#ifndef __ARITHMETIC_TUPLE_H__
#define __ARITHMETIC_TUPLE_H__

#include <vector>
#include <type_traits>
#include <array>
#include <tuple>

#include "rtc.h"
#include "exception.h"

_RTC_BEGIN
    // CLASS TEMPLATE arithmetic_tuple
template <typename T,
    size_t N>
    class arithmetic_tuple {
    protected:
        std::array<T, N> values;
    
    public:
        arithmetic_tuple() 
            : values() {}

        template <typename... Type,
            typename = std::enable_if_t<are_arithmetic<Type...>::value && sizeof...(Type) == N>>
            arithmetic_tuple(Type... vals) {
            values = { static_cast<T>(vals)... };
        }

        arithmetic_tuple(const T *arr) 
            : values() {
            for (size_t i = 0; i < N; ++i)
                values[i] = arr[i];
        }

        arithmetic_tuple(const arithmetic_tuple<T, N>& other) 
            : values(other.values) {}

        arithmetic_tuple<T, N>& operator=(const arithmetic_tuple<T, N>& other) {
            values = other.values;
            return *this;
        }

    template <typename P,
        typename = std::enable_if_t<std::is_convertible<P, T>::value>>
        operator arithmetic_tuple<P, N>() const {
            arithmetic_tuple<P, N> ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = (P)values[i];
            return ret;
        }

        T operator[](size_t idx) const {
            if (idx >= N)
                throw IndexOutOfBounds();
            return values[idx];
        }

        T& operator[](size_t idx) {
            if (idx >= N)
                throw IndexOutOfBounds();
            return values[idx];
        }

        friend bool operator==(const arithmetic_tuple<T, N>& t1,
            const arithmetic_tuple<T, N>& t2) {
            for (size_t i = 0; i < N; ++i)
                if (t1[i] != t2[i])
                    return false;
            return true;
        }

        friend bool operator!=(const arithmetic_tuple<T, N>& t1,
            const arithmetic_tuple<T, N>& t2) {
            for (size_t i = 0; i < N; ++i)
                if (t1[i] != t2[i])
                    return true;
            return false;
        }

        friend bool operator>=(const arithmetic_tuple<T, N>& t1,
            const arithmetic_tuple<T, N>& t2) {
            for (size_t i = 0; i < N; ++i)
                if (t1[i] < t2[i])
                    return false;
            return true;
        }

        friend bool operator<=(const arithmetic_tuple<T, N>& t1,
            const arithmetic_tuple<T, N>& t2) {
            for (size_t i = 0; i < N; ++i)
                if (t1[i] > t2[i])
                    return false;
            return true;
        }

        friend bool operator>(const arithmetic_tuple<T, N>& t1,
            const arithmetic_tuple<T, N>& t2) {
            for (size_t i = 0; i < N; ++i)
                if (t1[i] <= t2[i])
                    return false;
            return true;
        }

        friend bool operator<(const arithmetic_tuple<T, N>& t1,
            const arithmetic_tuple<T, N>& t2) {
            for (size_t i = 0; i < N; ++i)
                if (t1[i] >= t2[i])
                    return false;
            return true;
        }

        friend arithmetic_tuple<T, N> operator+(const arithmetic_tuple<T, N>& t1,
            const T& n) {
            arithmetic_tuple<T, N> ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = t1[i] + n;
            return ret;
        }

        friend arithmetic_tuple<T, N> operator+(const T& n,
            const arithmetic_tuple<T, N>& t2) {
            return t2 + n;
        }
            
        friend arithmetic_tuple<T, N> operator+(const arithmetic_tuple<T, N>& t1,
            const arithmetic_tuple<T, N>& t2) {
            arithmetic_tuple<T, N> ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = t1[i] + t2[i];
            return ret;
        }

        friend arithmetic_tuple<T, N> operator-(const arithmetic_tuple<T, N>& t1,
            const T& n) {
            arithmetic_tuple<T, N> ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = t1[i] - n;
            return ret;
        }

        friend arithmetic_tuple<T, N> operator-(const T& n,
            const arithmetic_tuple<T, N>& t1) {
            arithmetic_tuple<T, N> ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = n - t1[i];
            return ret;
        }

        friend arithmetic_tuple<T, N> operator-(const arithmetic_tuple<T, N>& t1,
            const arithmetic_tuple<T, N>& t2) {
            arithmetic_tuple<T, N> ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = t1[i] - t2[i];
            return ret;
        }

        friend arithmetic_tuple<T, N> operator*(const arithmetic_tuple<T, N>& t1,
            const T& n) {
            arithmetic_tuple<T, N> ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = t1[i] * n;
            return ret;
        }

        friend arithmetic_tuple<T, N> operator/(const arithmetic_tuple<T, N>& t1,
            const T& n) {
            arithmetic_tuple<T, N> ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = t1[i] / n;
            return ret;
        }

        friend arithmetic_tuple<T, N> operator*(const T& n,
            const arithmetic_tuple<T, N>& t1) {
            return t1 * n;
        }

        friend arithmetic_tuple<T, N> operator*(const arithmetic_tuple<T, N>& t1,
            const arithmetic_tuple<T, N>& t2) {
            arithmetic_tuple<T, N> ret;
            for (size_t i = 0; i < N; ++i)
                ret[i] = t1[i] * t2[i];
            return ret;
        }
    };
_RTC_END

#endif /* __ARITHMETIC_TUPLE_H__ */