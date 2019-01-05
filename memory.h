#ifndef __MEMORY_H__
#define __MEMORY_H__
#pragma _CRT_WARNING("remove memory.h from usage")
#include "defs.h"

#include <iostream>

#define HASH_SIZE 512
#define HASH_MASK (HASH_SIZE - 1)
#define HASH(obj) ((ptrdiff_t)(obj) & HASH_MASK)

_RTC_BEGIN
    class _Ptr_handler {
    private:
        size_t refs; //TODO: assuming size_t has atomic ++ and --
        const void* ptr;
    public:
        _DEVHOST explicit _Ptr_handler(const void* ptr)
            : ptr(ptr), refs(0) {}

        _DEVHOST size_t get_refs() const noexcept {
            return (refs);
        }

        _DEVHOST const void* get_ptr() const noexcept {
            return (ptr);
        }

        _DEVHOST size_t add_ref() noexcept {
            return refs++;
        }

        _DEVHOST size_t remove_ref() noexcept {
            return refs--;
        }
    };

    _array<_vector<_Ptr_handler>, HASH_SIZE> _Handlers;

template <class _Ty>
    class shared_ptr {
    private:
        using value_type = _Ty;

        _Ty *val;
        _Ptr_handler *handler;

        _DEVHOST _Ptr_handler * handler_lookup() const {
            _vector<_Ptr_handler>& v = _Handlers[HASH(val)];
            _Ptr_handler *handler = nullptr;

            for (size_t i = 0; i < v.size(); ++i) {
                if (v[i].get_ptr() != (void*)(val))
                    continue;
                handler = &v[i];
                break;
            }

            if (handler == nullptr) {
                v.emplace_back((void*)(val));
                handler = &v.back();
                //std::cout << "first shared pointer created for an object" << std::endl;
            }

            return handler;
        }

    public:
    //template <typename... _Args>
    //std::is_constructible<_Ty, _Args...>::value
        _DEVHOST shared_ptr(const _Ty& _Val) {
            val = new _Ty(_Val);
            handler = handler_lookup();
            handler->add_ref();
        }

        _DEVHOST shared_ptr(_Ty* _Val_ptr)
            : val(_Val_ptr) {
            handler = handler_lookup();
            handler->add_ref();
        }

        _DEVHOST shared_ptr<_Ty>& operator=(_Ty* _Val_ptr) {
            val = _Val_ptr;
            handler = handler_lookup();
            handler->add_ref();
        }

        _DEVHOST shared_ptr<_Ty>& operator=(const _Ty& _Val) {
            val = new _Ty(_Val);
            handler = handler_lookup();
            handler->add_ref();
        }

        _DEVHOST shared_ptr(const shared_ptr<_Ty>& other)
            : val(other.val), handler(other.handler) {
            handler->add_ref();
        }

        _DEVHOST shared_ptr<_Ty>& operator=(const shared_ptr<_Ty>& other) {
            val = other.val;
            handler = other.handler;
            handler->add_ref();
            return *this;
        }

        _DEVHOST ~shared_ptr() {
            //assert(handler != nullptr);
            if (handler->remove_ref() == 1) {
                //std::cout << "last shared pointer deleted" << std::endl;
                _vector<_Ptr_handler>& v = _Handlers[HASH(val)];
                for (size_t i = 0; i < v.size(); ++i)
                    if (&v[i] == handler)
                        v.erase(v.begin() + i);
                delete val;
            } /*else {
                std::cout << "shared pointer deleted" << std::endl;
            }*/
        }

        _DEVHOST _Ty* operator->() {
            return val;
        }

        _DEVHOST _Ty* operator->() const {
            return val;
        }

        _DEVHOST _Ty& operator*() {
            return *val;
        }

        _DEVHOST _Ty& operator*() const {
            return *val;
        }
    };
_RTC_END

#endif /* __MEMORY_H__ */