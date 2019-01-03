#ifndef __VECTOR_H__
#define __VECTOR_H__

#include "rtc.h"
#include "arithmetic_tuple.h"
#include "exception.h"

_RTC_BEGIN
    // ALIAS TEMPLATE vector
template <size_t N>
    using vector = arithmetic_tuple<vec_type, N>;

    using vec3 = vector<3>;

    using vec2 = vector<2>;

    // FUNCTION TEMPLATE dot
template <size_t N>
    vec_type dot(const vector<N>& v1,
        const vector<N>& v2) {
        vec_type ret = 0;
        for (size_t i = 0; i < N; ++i)
            ret += v1[i] * v2[i];
        return ret;
    }

    // FUNCTION TEMPLATE len2
template <size_t N>
    vec_type len2(const vector<N>& v1) {
        return dot(v1, v1);
    }

    // FUNCTION TEMPLATE len
template <size_t N>
    vec_type len(const vector<N>& v1) {
        return sqrt(len2(v1));
    }

    // FUNCTION TEMPLATE dist2
template <size_t N>
    vec_type dist2(const vector<N>& v1,
        const vector<N>& v2) {
        return len2(v1 - v2);
    }

    // FUNCTION TEMPLATE dist
template <size_t N>
    vec_type dist(const vector<N>& v1,
        const vector<N>& v2) {
        return len(v1 - v2);
    }

    // FUNCTION TEMPLATE cross
template <size_t N>
    vector<N> cross(const vector<N>& v1,
        const vector<N>& v2) {
        throw std::exception("unimplemented function");
    }

    vec3 cross(const vec3& v1,
        const vec3& v2) {
        return vec3(
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]);
    }

    // CLASS TEMPLATE normal
template <size_t N>
    class normal : public vector<N> {
    private:

        using _Mybase = vector<N>;

    template <typename... P>
        void normalize(P... args) {
            if (sizeof...(P) == 0)
                return;
            vec_type ln = len(*this);
            if (ln == 0)
                throw ZeroNormalVector();
            for (size_t i = 0; i < N; ++i)
                this->values[i] /= ln;
        }

    public:
        normal() 
            : _Mybase(unit_vector<N>(0)) {}

    template <typename... P>
        normal(P... args) 
            : _Mybase(args...) {
            normalize(args...);
        }
    };

    using normal3 = normal<3>;

    // FUNCTION TEMPLATE rotate
template <size_t N>
    vector<N> rotate(vector<N> v,
        normal<N> axis) {
        throw std::exception("unimplemented function");
    }

    // FUNCTION TEMPLATE reflect
template <size_t N>
    normal<N> reflect(const vector<N>& i,
        const normal<N>& n) {
        return i - (static_cast<vec_type>(2) * dot(n, i) * n);
    }

    // FUNCTION TEMPLATE unit_vector
template <size_t N>
    normal<N> unit_vector(size_t i) {
        vector<N> ret;
        ret[i] = 1;
        return ret;
    }

    // FUNCTION TEMPLATE standard_base
template <size_t N>
    std::array<normal<N>, N> standard_base() {
        std::array<normal<N>, N> ret;
        for (size_t i = 0; i < N; ++i)
            ret[i] = unit_vector<N>(i);
        return ret;
    }

    // FUNCTION TEMPLATE from_value
template <size_t N>
    vector<N> from_value(const vec_type& val) {
        vector<N> ret;
        for (size_t i = 0; i < N; ++i)
            ret[i] = val;
        return ret;
    }
_RTC_END

#endif /* __VECTOR_H__ */