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
     * @param idx_diag The index of diagonal.
     * @param idx_tar The index of global target.
     * @param size_a The size of the first array.
     * @param size_b The size of the second array.
     * @param p_crs_x [Out] The pointer to the x-axis coord of crossing point.
     * @param p_crs_y [Out] The pointer to the y-axis coord of crossing point.
     * @param p_is_a [Out] The pointer to indicate whether crossing point is taking value in a or b.
     */
    template <typename T>
    __device__ void _unit_merge(T* arr_tar, T* arr_a, T* arr_b, size_t idx_diag, size_t idx_tar, size_t size_a, size_t size_b,
                                int* p_crs_x=nullptr, int* p_crs_y=nullptr, bool* p_is_a=nullptr);


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
    __device__ void _unit_merge(T* arr_tar, T* arr_a, T* arr_b, size_t idx_diag, size_t idx_tar, size_t size_a, size_t size_b,
                                int* p_crs_x, int* p_crs_y, bool* p_is_a)
    {
        // assert(size_a >= size_b);
        if (idx_diag >= size_a+size_b) 
        {
            if (p_crs_x) { *p_crs_x = size_b; }
            if (p_crs_y) { *p_crs_y = size_a; }
            return ;
        }

        // The 0th coord is x, and 1st is y.
        int ll_x = 0, ll_y = 0, tr_x = 0, tr_y = 0, crt_x=0, crt_y=0;
        // size_t low_left[] = {0, 0}, top_right[] = {0, 0}, crt[] = {0, 0}; // Coords.
        // if (idx_diag > size_a)
        // {
        //     ll_x = idx_diag - size_a;
        //     ll_y = size_a;
        //     tr_x = size_a;
        //     tr_y = idx_diag - size_a;
        // }
        // else
        // {
        //     ll_x = 0;
        //     ll_y = idx_diag;
        //     tr_x = idx_diag;
        //     tr_y = 0;
        // }

        if (idx_diag > size_a)
        {
            ll_x = idx_diag - size_a;
            ll_y = size_a;
        }
        else
        {
            ll_x = 0;
            ll_y = idx_diag;
        }
        if (idx_diag > size_b)
        {
            tr_x = size_b;
            tr_y = idx_diag - size_b;
        }
        else
        {
            tr_x = idx_diag;
            tr_y = 0;
        }

        while( tr_x>=ll_x && tr_y<=ll_y )
        {
            size_t offset = (ll_y - tr_y) / 2;
            crt_x = ll_x + offset;
            crt_y = ll_y - offset;
            // if (crt_y>=0 && crt_x<=size_b) // Current coord in valid area.
            if (crt_x<=size_b) // Current coord in valid area.
            {
                if (crt_y==size_a || crt_x==0 || arr_a[crt_y]>arr_b[crt_x-1]) // l-l in 0 area.
                {
                    if (crt_x==size_b || crt_y==0 || arr_a[crt_y-1]<=arr_b[crt_x]) // u-r in 1 area.
                    {
                        if (crt_y<size_a && (crt_x==size_b || arr_a[crt_y]<=arr_b[crt_x]) )
                        { 
                            arr_tar[idx_tar] = arr_a[crt_y]; 
                            if (p_is_a) { *p_is_a = true; }
                            // printf("idx: %2zu. coord: (%zu, %zu). a: %d\n", idx, crt_x, crt_y, arr_tar[idx]);
                        }
                        else
                        { 
                            arr_tar[idx_tar] = arr_b[crt_x]; 
                            if (p_is_a) { *p_is_a = false; }
                            // printf("idx: %2zu. coord: (%zu, %zu). b: %d\n", idx, crt_x, crt_y, arr_tar[idx]);
                        }
                        if (p_crs_x) { *p_crs_x = crt_x; }
                        if (p_crs_y) { *p_crs_y = crt_y; }
                        break;
                    }
                    else // Move the low-left coord.
                    {
                        ll_x = crt_x + 1;
                        ll_y = crt_y - 1;
                    }
                }
                else // Move the top-right coord.
                {
                    tr_x = crt_x - 1;
                    tr_y = crt_y + 1;
                }
            }
            else // Move the rop-right coord.
            {
                tr_x = crt_x - 1;
                tr_y = crt_y + 1;
            }
        }
    }


    template <typename T>
    __global__ void merge_small_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b)
    {
        __shared__ T sarr_[1024];
        size_t idx = threadIdx.x;
        if (idx < size_a)
        { sarr_[idx] = arr_a[idx]; }
        else
        { sarr_[idx] = arr_b[idx-size_a]; }
        __syncthreads();

        _unit_merge<T>(arr_tar, sarr_, sarr_+size_a, idx, idx, size_a, size_b); 
        // _unit_merge<T>(arr_tar, arr_a, arr_b, idx, idx, size_a, size_b); 
        // if (size_a >= size_b)
        // { _unit_merge<T>(arr_tar, arr_a, arr_b, idx, size_a, size_b); }
        // else
        // { _unit_merge<T>(arr_tar, arr_b, arr_a, idx, size_b, size_a); }
    }
}
