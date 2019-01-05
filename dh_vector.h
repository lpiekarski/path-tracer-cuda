#ifndef __DH_VECTOR_H__
#define __DH_VECTOR_H__

_RTC_BEGIN
#define DH_VEC_DEFAULT_LENGTH 1
#define DH_VEC_SCALE_UP 2
#define DH_VEC_SCALE_DOWN 4
//TODO: make dh_vector thread safe
//TODO: add device_ctr
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

        _DEVHOST void resize(size_t new_size) {
            _Ty* new_arr = new _Ty[new_size];
            size_t copy_length = new_size;
            if (copy_length > sz)
                copy_length = sz;
            for (size_t i = 0; i < copy_length; ++i)
                new_arr[i] = arr[i];
            delete[] arr;
            arr_length = new_size;
            if (sz > arr_length)
                sz = arr_length;
            arr = new_arr;
        }

        //TODO: make thread safe
        _DEVHOST void push_back(const _Ty& _Elem) {
            if (sz == arr_length) {
                _Ty* new_arr = new _Ty[arr_length * DH_VEC_SCALE_UP];
                for (size_t i = 0; i < sz; ++i)
                    new_arr[i] = arr[i];
                delete[] arr;
                arr_length *= DH_VEC_SCALE_UP;
                arr = new_arr;
            }
            arr[sz++] = _Elem;
        }

        //TODO: make thread safe
        _DEVHOST void erase(size_t idx) {
            _Ty* new_arr;
            if ((sz - 1) * DH_VEC_SCALE_DOWN < arr_length)
                arr_length /= DH_VEC_SCALE_UP;
            new_arr = new _Ty[arr_length];
            size_t j = 0;
            for (size_t i = 0; i < sz; ++i) {
                if (i == idx)
                    continue;
                new_arr[j++] = arr[i];
            }
            delete[] arr;
            arr = new_arr;
        }

        _DEVHOST size_t begin() const noexcept {
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