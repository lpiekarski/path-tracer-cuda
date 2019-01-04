#ifndef __DH_VECTOR_H__
#define __DH_VECTOR_H__

_RTC_BEGIN
#define DH_VEC_DEFAULT_LENGTH 1
#define DH_VEC_SCALE_UP 2
#define DH_VEC_SCALE_DOWN 4
//TODO: make dh_vector thread safe
template <typename _Ty>
    class dh_vector {
    private:
        using value_type = _Ty;

        _Ty* arr;
        size_t sz, arr_length;
    public:
        _DEVHOST dh_vector() {
            arr = new _Ty[DH_VEC_DEFAULT_LENGTH];
            arr_length = DH_VEC_DEFAULT_LENGTH;
            sz = 0;
        }

        _DEVHOST dh_vector(const dh_vector& other) {
            arr = new _Ty[other.arr_length];
            arr_length = other.arr_length;
            sz = other.sz;
            for (size_t i = 0; i < sz; ++i)
                arr[i] = other.arr[i];
        }

        _DEVHOST ~dh_vector() {
            delete[] arr;
        }

        _DEVHOST void push_back(const _Ty& _Elem) {
            if (sz + 1 == arr_length) {

            }
            arr[sz++] = _Elem;
        }

        _DEVHOST constexpr size_t begin() const noexcept {
            return 0;
        }

        _DEVHOST size_t size() const noexcept {
            return sz;
        }

        _DEVHOST _Ty operator[](size_t idx) const {
            return arr[idx];
        }

        _DEVHOST _Ty& operator[](size_t idx) {
            return arr[idx];
        }
    };
_RTC_END

#endif /* __DH_VECTOR_H__ */