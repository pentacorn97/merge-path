/**
 * @file alg.cuh
 * @author  bx.zh
 * @brief  Declares the algorithms
 * @date 2022-03-05
 * 
 * @copyright Copyleft (c) 2022 all wrongs reserved.
 * 
 */
#pragma once
#include <cstdio>
#include <algorithm>
#include <assert.h>

namespace merge
{
    /**
     * @brief The function used to merge the specified diaganal.
     * 
     * @tparam T Type of data.
     * @param arr_tar The target array.
     * @param arr_a The first array, which is supposed to have larger size. 
     * @param arr_b The second array.
     * @param idx The index of diagonal.
     * @param size_a The size of the first array.
     * @param size_b The size of the second array.
     */
    template <typename T>
    __device__ void _unit_merge(T* arr_tar, T* arr_a, T* arr_b, size_t idx, size_t size_a, size_t size_b);


    /**
     * @brief The function used to merge small tables.
     * 
     * @tparam T Type of data.
     * @param arr_tar The target array.
     * @param arr_a The first array. 
     * @param arr_b The second array.
     * @param size_a The size of the first array.
     * @param size_b The size of the second array.
     */
    template <typename T>
    __global__ void merge_small_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b);



    /////////////////////////////////////////////////////////////
    // Implementation
    /////////////////////////////////////////////////////////////
    template <typename T>
    __device__ void _unit_merge(T* arr_tar, T* arr_a, T* arr_b, size_t idx, size_t size_a, size_t size_b)
    {
        // assert(size_a >= size_b);
        if (idx >= size_a+size_b) {return ;}

        // The 0th coord is x, and 1st is y.
        size_t low_left[] = {0, 0}, top_right[] = {0, 0}, crt[] = {0, 0}; // Coords.
        if (idx > size_a)
        {
            low_left[0] = idx - size_a;
            low_left[1] = size_a;
            top_right[0] = size_a;
            top_right[1] = idx - size_a;
        }
        else
        {
            low_left[0] = 0;
            low_left[1] = idx;
            top_right[0] = idx;
            top_right[1] = 0;
        }


        while( top_right[0]>=low_left[0] && top_right[1]<=low_left[1] )
        {
            size_t offset = (low_left[1] - top_right[1]) / 2;
            crt[0] = low_left[0] + offset;
            crt[1] = low_left[1] - offset;
            // if (crt[1]>=0 && crt[0]<=size_b) // Current coord in valid area.
            if (crt[0]<=size_b) // Current coord in valid area.
            {
                if (crt[1]==size_a || crt[0]==0 || arr_a[crt[1]]>arr_b[crt[0]-1]) // l-l in 0 area.
                {
                    if (crt[0]==size_b || crt[1]==0 || arr_a[crt[1]-1]<=arr_b[crt[0]]) // u-r in 1 area.
                    {
                        if (crt[1]<size_a && (crt[0]==size_b || arr_a[crt[1]]<=arr_b[crt[0]]) )
                        { 
                            arr_tar[idx] = arr_a[crt[1]]; 
                            // printf("idx: %2zu. coord: (%zu, %zu). a: %d\n", idx, crt[0], crt[1], arr_tar[idx]);
                        }
                        else
                        { 
                            arr_tar[idx] = arr_b[crt[0]]; 
                            // printf("idx: %2zu. coord: (%zu, %zu). b: %d\n", idx, crt[0], crt[1], arr_tar[idx]);
                        }
                        break;
                    }
                    else // Move the low-left coord.
                    {
                        low_left[0] = crt[0] + 1;
                        low_left[1] = crt[1] - 1;
                    }
                }
                else // Move the top-right coord.
                {
                    top_right[0] = crt[0] - 1;
                    top_right[1] = crt[1] + 1;
                }
            }
            else // Move the rop-right coord.
            {
                top_right[0] = crt[0] - 1;
                top_right[1] = crt[1] + 1;
            }
        }
    }


    template <typename T>
    __global__ void merge_small_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b)
    {
        size_t idx = threadIdx.x;
         _unit_merge<T>(arr_tar, arr_a, arr_b, idx, size_a, size_b); 
        // if (size_a >= size_b)
        // { _unit_merge<T>(arr_tar, arr_a, arr_b, idx, size_a, size_b); }
        // else
        // { _unit_merge<T>(arr_tar, arr_b, arr_a, idx, size_b, size_a); }
    }
}
