#ifndef __EXCEPTION_H__
#define __EXCEPTION_H__

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
_RTC_END

#endif /* __EXCEPTION_H__ */