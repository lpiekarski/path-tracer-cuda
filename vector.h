#ifndef __VECTOR_H__
#define __VECTOR_H__

#include "rtc.h"
#include "arithmetic_tuple.h"
#include "exception.h"

_RTC_BEGIN
    // ALIAS TEMPLATE vector
template <size_t _Dims>
    using vector = arithmetic_tuple<vec_type, _Dims>;

    using vec3 = vector<3>;

    using vec2 = vector<2>;

    // FUNCTION TEMPLATE dot
template <size_t _Dims>
    _DEVHOST vec_type dot(const vector<_Dims>& v1,
        const vector<_Dims>& v2) {
        vec_type ret = 0;
        for (size_t i = 0; i < _Dims; ++i)
            ret += v1[i] * v2[i];
        return ret;
    }

    // FUNCTION TEMPLATE len2
template <size_t _Dims>
    _DEVHOST vec_type len2(const vector<_Dims>& v1) {
        return dot(v1, v1);
    }

    // FUNCTION TEMPLATE len
template <size_t _Dims>
    _DEVHOST vec_type len(const vector<_Dims>& v1) {
        return sqrt(len2(v1));
    }

    // FUNCTION TEMPLATE dist2
template <size_t _Dims>
    _DEVHOST vec_type dist2(const vector<_Dims>& v1,
        const vector<_Dims>& v2) {
        return len2(v1 - v2);
    }

    // FUNCTION TEMPLATE dist
template <size_t _Dims>
    _DEVHOST vec_type dist(const vector<_Dims>& v1,
        const vector<_Dims>& v2) {
        return len(v1 - v2);
    }

    // FUNCTION TEMPLATE cross
template <size_t _Dims>
    _DEVHOST vector<_Dims> cross(const vector<_Dims>& v1,
        const vector<_Dims>& v2) {
        throw_exc(std::exception("unimplemented function"));
    }

    _DEVHOST vec3 cross(const vec3& v1,
        const vec3& v2) {
        return vec3(
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]);
    }

    // CLASS TEMPLATE normal
template <size_t _Dims>
    class normal : public vector<_Dims> {
    private:
        using _Mybase = vector<_Dims>;

    template <typename... P>
        _DEVHOST void normalize(P... args) {
            if (sizeof...(P) == 0)
                return;
            vec_type ln = len(*this);
            //if (ln == 0)
            //    throw_exc(ZeroNormalVector());
            for (size_t i = 0; i < _Dims; ++i)
                this->values[i] /= ln;
        }

    public:
        _DEVHOST normal()
            : _Mybase(unit_vector<_Dims>(0)) {}

    template <typename... P>
        _DEVHOST normal(P... args) 
            : _Mybase(args...) {
            normalize(args...);
        }
    };

    using normal3 = normal<3>;

    // FUNCTION TEMPLATE rotate
template <size_t _Dims>
    _DEVHOST vector<_Dims> rotate(vector<_Dims> v,
        normal<_Dims> axis) {
        throw_exc(std::exception("unimplemented function"));
    }

    // FUNCTION TEMPLATE reflect
template <size_t _Dims>
    _DEVHOST normal<_Dims> reflect(const vector<_Dims>& i,
        const normal<_Dims>& n) {
        return i - (static_cast<vec_type>(2) * dot(n, i) * n);
    }

    // FUNCTION TEMPLATE unit_vector
template <size_t _Dims>
    _DEVHOST normal<_Dims> unit_vector(size_t i) {
        vector<_Dims> ret;
        ret[i] = 1;
        return ret;
    }

    // FUNCTION TEMPLATE standard_base
template <size_t _Dims>
    _DEVHOST _array<normal<_Dims>, _Dims> standard_base() {
        _array<normal<_Dims>, _Dims> ret;
        for (size_t i = 0; i < _Dims; ++i)
            ret[i] = unit_vector<_Dims>(i);
        return ret;
    }

    // FUNCTION TEMPLATE from_value
template <size_t _Dims>
    _DEVHOST vector<_Dims> from_value(const vec_type& val) {
        vector<_Dims> ret;
        for (size_t i = 0; i < _Dims; ++i)
            ret[i] = val;
        return ret;
    }
_RTC_END

#endif /* __VECTOR_H__ */