#ifndef __BITMAP_H__
#define __BITMAP_H__

#include <fstream>

#include "defs.h"
#include "rtc.h"
#include "exception.h"
#include "color.h"

_RTC_BEGIN
    // CLASS TEMPLATE bitmap
template <size_t BPP>
    class bitmap {
    private:
        _vector<char> data;
        size_t row_size;
        size_t width, height;

    public:
        using _Mytype = bitmap<BPP>;

        _DEVHOST bitmap(size_t width, size_t height)
            : data(), 
            width(width),
            height(height) {
            row_size = (BPP * 8 * width + 31) / 32 * 4;
            data.resize(14 + 40 + row_size * height);
            // bitmap file header
                //file format bytes
            data[0] = 'B';  
            data[1] = 'M';
                // file size bytes
            little_endian_insert<size_t>(data.size(), 4, data, 2); 
                // offset
            little_endian_insert<size_t>(14 + 40, 4, data, 10); 

            // BITMAPINFOHEADER 
                // the size of this header
            little_endian_insert<size_t>(40, 4, data, 14); 
                // the bitmap width in pixels
            little_endian_insert<size_t>(width, 4, data, 18); 
                // the bitmap height in pixels
            little_endian_insert<size_t>(height, 4, data, 22); 
                // the number of color planes
            little_endian_insert<size_t>(1, 2, data, 26); 
                // the number of bits per pixel
            little_endian_insert<size_t>(BPP * 8, 2, data, 28); 
                // horizontal resolution
            little_endian_insert<size_t>(2835, 4, data, 38);
                // vertical resolution
            little_endian_insert<size_t>(2835, 4, data, 42); 
        }

        _HOST bitmap(const char* filename) { read(filename); }

        _DEVHOST bitmap(const bitmap& other) {
            data = other.data;
            row_size = other.row_size;
            width = other.width;
            height = other.height;
        }

        _DEVHOST color<BPP> get(size_t x, size_t y) const {
            size_t bytes = BPP;
            size_t off = 14 + 40 + y * row_size + x * bytes;
            if (off + bytes - 1 >= data.size())
                throw IndexOutOfBounds();

            byte_color<BPP> ret;
            for (size_t i = 0; i < bytes; ++i)
                ret[i] = data[off + bytes - 1 - i];

            return color<BPP>(ret);
        }

        _DEVHOST void set(size_t x, size_t y, const color<BPP>& val) {
            byte_color<BPP> v(val);
            size_t bytes = BPP;
            size_t off = 14 + 40 + y * row_size + x * bytes;
            if (off + bytes - 1 >= data.size())
                throw IndexOutOfBounds();

            for (size_t i = 0; i < bytes; ++i)
                data[off + bytes - 1 - i] = v[i];
        }

        _HOST bool write(const char *filename) const {
            std::ofstream of(filename, std::ios::binary);
            if (!of)
                return false;
            char *buf = new char[data.size()];
            for (size_t i = 0; i < data.size(); ++i)
                buf[i] = data[i];
            of.write(buf, data.size());
            of.close();
            delete[] buf;
            return true;
        }
        
        _HOST bool read(const char *filename) {
            std::ifstream f(filename, std::ios::binary);
            if (!f)
                return false;
            f.seekg(0, f.end);
            size_t sz = f.tellg();
            f.seekg(0);
            char *buf = new char[sz];
            f.read(buf, sz);
            data.clear();
            for (size_t i = 0; i < sz; ++i)
                data.push_back(buf[i]);
            f.close();
            width = little_endian_read<size_t>(4, data, 18);
            height = little_endian_read<size_t>(4, data, 22);
            row_size = ((BPP * 8 * width + 31) / 32 * 4);

            delete[] buf;
            return true;
        }

#ifdef RTC_USE_CUDA
        template <typename... _Args>
        //typename = enable_if_t<is_constructible<_Mytype, _Args...>::value>
        _HOST static _Mytype* device_ctr(_Args... args) {
            _Mytype h_ret(args...);
            _Mytype *d_ret_ptr;
            size_t sz = sizeof(_Mytype);
            cudaMalloc((void**)&d_ret_ptr, sz);
            cudaMemcpy(d_ret_ptr, &h_ret, sz, cudaMemcpyHostToDevice);
            return d_ret_ptr;
        }
#endif /* RTC_USE_CUDA */
    };
_RTC_END

#endif /* __BITMAP_H__ */