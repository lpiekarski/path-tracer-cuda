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
    static constexpr int COL_LOG_RADIX = 8;
    static constexpr int COL_RADIX = 1 << COL_LOG_RADIX;
    static constexpr int COL_RADIX_MASK = COL_RADIX - 1;

    // ALIAS TEMPLATE byte_color
template <size_t BPP>
    using byte_color = arithmetic_tuple<char, BPP>;

    using col24bpp = byte_color<3>;

    using col32bpp = byte_color<4>;

    std::ostream& operator<<(std::ostream& s, col24bpp& c) {
        s << '#';
        s << std::setbase(16) << std::setw(6) << std::setfill('0') << 
            ((((c[0] << COL_LOG_RADIX) + c[1]) << COL_LOG_RADIX) + c[2]);
        return s;
    }

    std::istream& operator>>(std::istream& s, col24bpp& c) {
        s >> c[0] >> c[1] >> c[2];
        return s;
    }
    
    // FUNCTION TEMPLATE operator*
template <size_t BPP>
    byte_color<BPP> operator*(const byte_color<BPP>& c1, const byte_color<BPP>& c2) {
        byte_color<BPP> ret;
        size_t N = BPP;
        for (size_t i = 0; i < N; ++i)
            ret[i] = static_cast<char>(static_cast<uint16_t>(c1[i]) * static_cast<uint16_t>(c2[i]) / 255);
        return ret;
    }

    // FUNCTION TEMPLATE operator+
template <size_t BPP>
    byte_color<BPP> operator+(const byte_color<BPP>& c1, const byte_color<BPP>& c2) {
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
        using _Mybase =  arithmetic_tuple<vec_type, BPP>;

        char coord_to_char(const vec_type& x) const {
            return (char)(((static_cast<vec_type>(2) / (exp(-x) + static_cast<vec_type>(1))) - static_cast<vec_type>(1)) * COL_RADIX);
        }

        vec_type char_to_coord(const char& x) const {
            return -log(static_cast<vec_type>(2) / (x / static_cast<vec_type>(COL_RADIX) + static_cast<vec_type>(1)) - static_cast<vec_type>(1));
        }

    public:
        color() : _Mybase() {}

        color(const vec_type *arr) : _Mybase(arr) {}

        color(const color<BPP>& other) :
            _Mybase(other.values.data()) {}

        color(const byte_color<BPP>& other) {
            size_t N = BPP;
            for (size_t i = 0; i < N; ++i)
                this->values[i] = char_to_coord(other[i]);
        }

        color(const _Mybase& other) : _Mybase(other) {}

        operator byte_color<BPP>() const {
            byte_color<BPP> ret;
            size_t N = BPP;
            for (size_t i = 0; i < N; ++i)
                ret[i] = coord_to_char(this->values[i]);
            return ret;
        }

    template <typename... Type,
        typename = std::enable_if_t<are_convertible<vec_type, Type...>::value && sizeof...(Type) == BPP>>
        color(Type... vals) : _Mybase() {
            this->values = { static_cast<vec_type>(vals)... };
        }

    };

    // FUNCTION TEMPLATE gray
template <size_t BPP>
    color<BPP> gray(const vec_type& val) {
        color<BPP> ret;
        size_t N = BPP;
        for (size_t i = 0; i < N; ++i)
            ret[i] = val;
        return ret;
    }

    // FUNCTION TEMPLATE white
template <size_t BPP>
    color<BPP> white() {
        return gray<BPP>(static_cast<vec_type>(1));
    }
_RTC_END

#endif /* __COLOR_H__ */