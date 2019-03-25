#ifndef __COLOR_H__
#define __COLOR_H__

#include <exception>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>

#include "rtc.h"
#include "arithmetic_tuple.h"

_RTC_BEGIN
    constexpr unsigned int COL_LOG_RADIX = 8;
    constexpr unsigned int COL_RADIX = 1 << COL_LOG_RADIX;
    constexpr unsigned int COL_RADIX_MASK = COL_RADIX - 1;

    // ALIAS TEMPLATE byte_color
template <size_t BPP>
    using byte_color = arithmetic_tuple<char, BPP>;

    using col24bpp = byte_color<3>;

    using col32bpp = byte_color<4>;

    _HOST std::ostream& operator<<(std::ostream& s, col24bpp& c) {
        s << '#';
        s << std::setbase(16) << std::setw(6) << std::setfill('0') << 
            ((((c[0] << COL_LOG_RADIX) + c[1]) << COL_LOG_RADIX) + c[2]);
        return s;
    }

    _HOST std::istream& operator>>(std::istream& s, col24bpp& c) {
        s >> c[0] >> c[1] >> c[2];
        return s;
    }
    
    // FUNCTION TEMPLATE operator*
template <size_t BPP>
    _DEVHOST byte_color<BPP> operator*(const byte_color<BPP>& c1, const byte_color<BPP>& c2) {
        byte_color<BPP> ret;
        size_t N = BPP;
        for (size_t i = 0; i < N; ++i)
            ret[i] = static_cast<char>(static_cast<uint16_t>(c1[i]) * static_cast<uint16_t>(c2[i]) / 255);
        return ret;
    }

    // FUNCTION TEMPLATE operator+
template <size_t BPP>
    _DEVHOST byte_color<BPP> operator+(const byte_color<BPP>& c1, const byte_color<BPP>& c2) {
        byte_color<BPP> ret;
        size_t N = BPP;
        for (size_t i = 0; i < N; ++i) {
            uint16_t val = static_cast<uint16_t>(c1[i]) + static_cast<uint16_t>(c2[i]);
            if (val > COL_RADIX_MASK)
                val = COL_RADIX_MASK;
            ret[i] = val;
        }
        return ret;
    }

    // CLASS TEMPLATE color
template <size_t BPP>
    class color : public arithmetic_tuple<vec_type, BPP> {
    private:
        _DEVHOST char coord_to_char(const vec_type& x) const {
            return (char)(((static_cast<vec_type>(2) / 
                (exp(-x) + static_cast<vec_type>(1))) - 
                static_cast<vec_type>(1)) * COL_RADIX);
        }

        _DEVHOST vec_type char_to_coord(const char& x) const {
            return -log(static_cast<vec_type>(2) / 
                (x / static_cast<vec_type>(COL_RADIX) + 
                static_cast<vec_type>(1)) - static_cast<vec_type>(1));
        }

    public:
        using _Mybase = arithmetic_tuple<vec_type, BPP>;
        using _Mytype = color<BPP>;

        _DEVHOST color() : _Mybase() {}

        _DEVHOST color(const vec_type *arr) : _Mybase(arr) {}

        _DEVHOST color(const color<BPP>& other) :
            _Mybase(other.values.data()) {}

        _DEVHOST color(const byte_color<BPP>& other) {
            size_t N = BPP;
            for (size_t i = 0; i < N; ++i)
                this->values[i] = char_to_coord(other[i]);
        }

        _DEVHOST color(const _Mybase& other) : _Mybase(other) {}

        _DEVHOST operator byte_color<BPP>() const {
            byte_color<BPP> ret;
            size_t N = BPP;
            for (size_t i = 0; i < N; ++i)
                ret[i] = coord_to_char(this->values[i]);
            return ret;
        }

    template <typename... _Args,
        typename = enable_if_t<are_convertible<vec_type, _Args...>::value && sizeof...(_Args) == BPP>>
        _DEVHOST color(_Args... vals) : _Mybase() {
            this->values = { static_cast<vec_type>(vals)... };
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
    };

    // FUNCTION TEMPLATE gray
template <size_t BPP>
    _DEVHOST color<BPP> gray(const vec_type& val) {
        color<BPP> ret;
        size_t N = BPP;
        for (size_t i = 0; i < N; ++i)
            ret[i] = val;
        return ret;
    }

    // FUNCTION TEMPLATE white
template <size_t BPP>
    _DEVHOST color<BPP> white() {
        return gray<BPP>(static_cast<vec_type>(1));
    }
_RTC_END

#endif /* __COLOR_H__ */