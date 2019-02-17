#ifndef __DH_VECTOR_H__
#define __DH_VECTOR_H__

_RTC_BEGIN
#define DH_VEC_DEFAULT_LENGTH 1
#define DH_VEC_SCALE_UP 2
#define DH_VEC_SCALE_DOWN 4

template <typename _Ty>
    class dh_vector {
    private:
        using value_type = _Ty;
        using _Mytype = dh_vector<_Ty>;

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

        _DEVHOST dh_vector& operator=(const dh_vector& other) {
            delete[] arr;
            arr = new _Ty[other.arr_length];
            arr_length = other.arr_length;
            sz = other.sz;
            for (size_t i = 0; i < sz; ++i)
                arr[i] = other.arr[i];
            return *this;
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
            sz = new_size;
            arr = new_arr;
            for (size_t i = copy_length; i < sz; ++i)
                arr[i] = 0;
        }

        _DEVHOST void clear() {
            sz = 0;
        }

        _DEVHOST _Ty* data() {
            return arr;
        }

        _DEVHOST const _Ty* data() const {
            return arr;
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
        //TODO test
#ifdef RTC_USE_CUDA
    template <typename... _Args,
        typename = enable_if_t<is_constructible<_Mytype, _Args...>::value>>
            _HOST static _Mytype* device_ctr(_Args... args) {
            _Mytype h_ret(args...);
            _Mytype *d_ret_ptr;
            size_t sz = sizeof(_Mytype);
            cudaMalloc((void**)&d_ret_ptr, sz);
            cudaMemcpy(d_ret_ptr, &h_ret, sz, cudaMemcpyHostToDevice);
            size_t arr_sz = sizeof(_Ty) * h_ret.size();
            _Ty *hostarr;
            cudaMalloc((void**)&hostarr, arr_sz);
            cudaMemcpy(hostarr, h_ret.arr, arr_sz, cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_ret_ptr->arr), &hostarr, sizeof(_Ty *), cudaMemcpyHostToDevice);
            return d_ret_ptr;
        }

        //TODO test deleting d_ptr->arr
        _HOST static void device_dtr(_Mytype *d_ptr) {
            cudaFree(&(d_ptr->arr));
            cudaFree(d_ptr);
        }

        _HOST static _Mytype host_cpy(_Mytype *d_ptr) {
            _Mytype ret;
            delete[] (ret.arr);
            size_t sz = sizeof(_Mytype);
            cudaMemcpy(&ret, d_ptr, sz, cudaMemcpyDeviceToHost);
            size_t arr_sz = sizeof(_Ty) * ret.size();
            _Ty *devarr;
            cudaMemcpy(&devarr, &(d_ptr->arr), sizeof(_Ty *), cudaMemcpyDeviceToHost);
            ret.arr = new _Ty[ret.size()];
            cudaMemcpy(ret.arr, devarr, arr_sz, cudaMemcpyDeviceToHost);
            return ret;
        }
#endif /* RTC_USE_CUDA */
    };
_RTC_END

#endif /* __DH_VECTOR_H__ */