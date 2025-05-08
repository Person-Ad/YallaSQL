#pragma once
#include "kernels/string_kernel.hpp"
#include "kernels/constants.hpp"
#include "utils/macros.hpp"
#include "enums/data_type.hpp"
#include <stdint-gcc.h>

namespace YallaSQL::Kernel
{
    template <typename T>
    void launch_cross_product_col(const T* __restrict__ col, T* __restrict__ out_col, const char*  __restrict__ src_nullset, char*  __restrict__ out_nullset, const uint32_t leftBs, const uint32_t rightBs, cudaStream_t stream, const bool isleft) ;

    inline void launch_cross_product_type(void* src, void* dist, char* src_nullset, char* dist_nullset, DataType type, const int lbs, const int rbs, cudaStream_t stream, bool isleft) {
        switch (type)
        {
        case DataType::INT:
            YallaSQL::Kernel::launch_cross_product_col(
                static_cast<int*>(src),
                static_cast<int*>(dist),
                src_nullset,
                dist_nullset,
                lbs,
                rbs,
                stream,
                isleft
            );
            break;
        case DataType::FLOAT:
            YallaSQL::Kernel::launch_cross_product_col(
                static_cast<float*>(src),
                static_cast<float*>(dist),
                src_nullset,
                dist_nullset,
                lbs,
                rbs,
                stream,
                isleft
            );
            break;
        case DataType::DATETIME:
            YallaSQL::Kernel::launch_cross_product_col(
                static_cast<int64_t*>(src),
                static_cast<int64_t*>(dist),
                src_nullset,
                dist_nullset,
                lbs,
                rbs,
                stream,
                isleft
            );
            break;

        case DataType::STRING:
            YallaSQL::Kernel::launch_cross_product_col(
                static_cast<YallaSQL::Kernel::String*>(src),
                static_cast<YallaSQL::Kernel::String*>(dist),
                src_nullset,
                dist_nullset,
                lbs,
                rbs,
                stream,
                isleft
            );
            break;
        
        default:
            break;
        }
    }

inline void launch_cross_product_one_type(void* src, void* dist, uint32_t offset, char* src_nullset, char* dist_nullset, DataType type, const int lbs, const int rbs, cudaStream_t stream, bool isleft) {
        switch (type)
        {
        case DataType::INT:
            YallaSQL::Kernel::launch_cross_product_col(
                static_cast<int*>(src) + offset,
                static_cast<int*>(dist),
                src_nullset + offset,
                dist_nullset,
                lbs,
                rbs,
                stream,
                isleft
            );
            break;
        case DataType::FLOAT:
            YallaSQL::Kernel::launch_cross_product_col(
                static_cast<float*>(src) + offset,
                static_cast<float*>(dist),
                src_nullset + offset,
                dist_nullset,
                lbs,
                rbs,
                stream,
                isleft
            );
            break;
        case DataType::DATETIME:
            YallaSQL::Kernel::launch_cross_product_col(
                static_cast<int64_t*>(src) + offset,
                static_cast<int64_t*>(dist),
                src_nullset + offset,
                dist_nullset,
                lbs,
                rbs,
                stream,
                isleft
            );
            break;

        case DataType::STRING:
            YallaSQL::Kernel::launch_cross_product_col(
                static_cast<YallaSQL::Kernel::String*>(src) + offset,
                static_cast<YallaSQL::Kernel::String*>(dist),
                src_nullset + offset,
                dist_nullset,
                lbs,
                rbs,
                stream,
                isleft
            );
            break;
        
        default:
            break;
        }
    }
}
