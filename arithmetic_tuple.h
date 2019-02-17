#ifndef __ARITHMETIC_TUPLE_H__
#define __ARITHMETIC_TUPLE_H__

#include "rtc.h"
#include "exception.h"

_RTC_BEGIN
    // CLASS TEMPLATE arithmetic_tuple
template <typename _Ty,
    size_t _Size>
    class arithmetic_tuple {
    protected:
        _array<_Ty, _Size> values;
    
    public:
        using value_type = _Ty;
        using _Mytype = arithmetic_tuple<_Ty, _Size>;

        _DEVHOST arithmetic_tuple()
            : values() {}

        template <typename... _Args,
            typename = enable_if_t<are_arithmetic<_Args...>::value && sizeof...(_Args) == _Size>>
            _DEVHOST arithmetic_tuple(_Args... args) {
            values = { static_cast<_Ty>(args)... };
        }

        _DEVHOST arithmetic_tuple(const _Ty *_Arr)
            : values() {
            for (size_t i = 0; i < _Size; ++i)
                values[i] = _Arr[i];
        }

        _DEVHOST arithmetic_tuple(const _Mytype& other)
            : values(other.values) {}

        _DEVHOST _Mytype& operator=(const _Mytype& other) {
            values = other.values;
            return *this;
        }

    template <typename _Conv,
        typename = enable_if_t<is_convertible<_Conv, _Ty>::value>>
        _DEVHOST operator arithmetic_tuple<_Conv, _Size>() const {
            arithmetic_tuple<_Conv, _Size> ret;
            for (size_t i = 0; i < _Size; ++i)
                ret[i] = (_Conv)values[i];
            return ret;
        }
#ifdef RTC_USE_CUDA
    template <typename... _Args,
        typename = enable_if_t<is_constructible<_Mytype, _Args...>::value>>
        _HOST static _Mytype* device_ctr(_Args... args) {
            _Mytype h_ret(args...);
            _Mytype *d_ret_ptr;
            size_t sz = sizeof(_Mytype);
            cudaMalloc((void**)&d_ret_ptr, sz);
            cudaMemcpy(d_ret_ptr, &h_ret, sz, cudaMemcpyHostToDevice);
            return d_ret_ptr;
        }

        _HOST static void device_dtr(_Mytype *ptr) {
                cudaFree(ptr);
        }

        _HOST static _Mytype host_cpy(_Mytype *d_ptr) {
            _Mytype ret;
            cudaMemcpy(&ret, d_ptr, sizeof(_Mytype), cudaMemcpyDeviceToHost);
            return ret;
        }
#endif /* RTC_USE_CUDA */
        _DEVHOST _Ty operator[](size_t idx) const {
            //if (idx >= _Size)
            //    throw_exc(IndexOutOfBounds());
            return values[idx];
        }

        _DEVHOST _Ty& operator[](size_t idx) {
            //if (idx >= _Size)
            //    throw_exc(IndexOutOfBounds());
            return values[idx];
        }

        _DEVHOST friend bool operator==(const _Mytype& t1,
            const _Mytype& t2) {
            for (size_t i = 0; i < _Size; ++i)
                if (t1[i] != t2[i])
                    return false;
            return true;
        }

        _DEVHOST friend bool operator!=(const _Mytype& t1,
            const _Mytype& t2) {
            for (size_t i = 0; i < _Size; ++i)
                if (t1[i] != t2[i])
                    return true;
            return false;
        }

        _DEVHOST friend bool operator>=(const _Mytype& t1,
            const _Mytype& t2) {
            for (size_t i = 0; i < _Size; ++i)
                if (t1[i] < t2[i])
                    return false;
            return true;
        }

        _DEVHOST friend bool operator<=(const _Mytype& t1,
            const _Mytype& t2) {
            for (size_t i = 0; i < _Size; ++i)
                if (t1[i] > t2[i])
                    return false;
            return true;
        }

        _DEVHOST friend bool operator>(const _Mytype& t1,
            const _Mytype& t2) {
            for (size_t i = 0; i < _Size; ++i)
                if (t1[i] <= t2[i])
                    return false;
            return true;
        }

        _DEVHOST friend bool operator<(const _Mytype& t1,
            const _Mytype& t2) {
            for (size_t i = 0; i < _Size; ++i)
                if (t1[i] >= t2[i])
                    return false;
            return true;
        }

        _DEVHOST friend _Mytype operator+(const _Mytype& t1,
            const _Ty& n) {
            _Mytype ret;
            for (size_t i = 0; i < _Size; ++i)
                ret[i] = t1[i] + n;
            return ret;
        }

        _DEVHOST friend _Mytype operator+(const _Ty& n,
            const _Mytype& t2) {
            return t2 + n;
        }
            
        _DEVHOST friend _Mytype operator+(const _Mytype& t1,
            const _Mytype& t2) {
            _Mytype ret;
            for (size_t i = 0; i < _Size; ++i)
                ret[i] = t1[i] + t2[i];
            return ret;
        }

        _DEVHOST friend _Mytype operator-(const _Mytype& t1,
            const _Ty& n) {
            _Mytype ret;
            for (size_t i = 0; i < _Size; ++i)
                ret[i] = t1[i] - n;
            return ret;
        }

        _DEVHOST friend _Mytype operator-(const _Ty& n,
            const _Mytype& t1) {
            arithmetic_tuple<_Ty, _Size> ret;
            for (size_t i = 0; i < _Size; ++i)
                ret[i] = n - t1[i];
            return ret;
        }

        _DEVHOST friend _Mytype operator-(const _Mytype& t1,
            const _Mytype& t2) {
            _Mytype ret;
            for (size_t i = 0; i < _Size; ++i)
                ret[i] = t1[i] - t2[i];
            return ret;
        }

        _DEVHOST friend _Mytype operator*(const _Mytype& t1,
            const _Ty& n) {
            _Mytype ret;
            for (size_t i = 0; i < _Size; ++i)
                ret[i] = t1[i] * n;
            return ret;
        }

        _DEVHOST friend _Mytype operator/(const _Mytype& t1,
            const _Ty& n) {
            _Mytype ret;
            for (size_t i = 0; i < _Size; ++i)
                ret[i] = t1[i] / n;
            return ret;
        }

        _DEVHOST friend _Mytype operator*(const _Ty& n,
            const _Mytype& t1) {
            return t1 * n;
        }

        _DEVHOST friend _Mytype operator*(const _Mytype& t1,
            const _Mytype& t2) {
            _Mytype ret;
            for (size_t i = 0; i < _Size; ++i)
                ret[i] = t1[i] * t2[i];
            return ret;
        }
    };
_RTC_END

#endif /* __ARITHMETIC_TUPLE_H__ */