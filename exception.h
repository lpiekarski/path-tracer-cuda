#ifndef __EXCEPTION_H__
#define __EXCEPTION_H__
/*
    throwing exceptions is currently 
    not supported in device code.
*/

#include <exception>

#include "rtc.h"

_RTC_BEGIN
    class IndexOutOfBounds : public std::exception {
    public:
        virtual char const * what() const noexcept {
            return "IndexOutOfBounds";
        }
    };

    class ZeroNormalVector : public std::exception {
    public:
        virtual char const * what() const noexcept {
            return "ZeroNormalVector";
        }
    };

    _DEVHOST void throw_exc(const std::exception& e) {
#ifndef __CUDA_ARCH__
        throw e;
#endif
    }

_RTC_END

#endif /* __EXCEPTION_H__ */