#pragma once
#include <cstdio>
#include <algorithm>
#include <assert.h>

#include "./alg.cuh"

namespace merge
{
    /**
     * @brief The function used to merge big (siza_a+size_b>1024) tables.
     * 
     * @tparam T Type of data.
     * @param arr_tar The target array.
     * @param arr_a The first array. 
     * @param arr_b The second array.
     * @param diag_idx The index of the diagnal.
     * @param size_a The size of the first array.
     * @param size_b The size of the second array.
     * @param crossing_x The x coordinate of the intersection between the path and the diag_idx -th cross-diagnal
     * @param crossing_y The y coordinate of the intersection between the path and the diag_idx -th cross-diagnal
     * @param a_or_b true if the value is taken at arr_a, else false
     */
    template <typename T>
    __device__ void _one_diag_idx(T* arr_tar, T* arr_a, T* arr_b, int diag_idx, size_t size_a, size_t size_b, int *crossing_x, int *crossing_y, bool *a_or_b);
    
    /**
     * @brief The function used to merge big (siza_a+size_b>1024) tables.
     * 
     * @tparam T Type of data.
     * @param arr_tar The target array.
     * @param arr_a The first array. 
     * @param arr_b The second array.
     * @param diag_idx The index of the diagnal.
     * @param tar_idx The index of the target array at which we should put the result.
     * @param size_a The size of the first array.
     * @param size_b The size of the second array.
     */
    template <typename T>
    __device__ void _one_diag(T* arr_tar, T* arr_a, T* arr_b, int diag_idx, int tar_idx, size_t size_a, size_t size_b);

    template <typename T>
    __global__ void merge_small_idx_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b);

    /**
     * @brief The function used to merge big (siza_a+size_b>1024) tables.
     * 
     * @tparam T Type of data.
     * @param arr_tar The target array.
     * @param arr_a The first array. 
     * @param arr_b The second array.
     * @param size_a The size of the first array.
     * @param size_b The size of the second array.
     */
    template <typename T>
    __global__ void merge_big_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b);


    /////////////////////////////////////////////////////////////
    // Implementation
    /////////////////////////////////////////////////////////////
    template <typename T>
    __device__ void _one_diag_idx(T* arr_tar, T* arr_a, T* arr_b, int diag_idx, size_t size_a, size_t size_b, int *crossing_x, int *crossing_y, bool* a_or_b)
    {
        if (diag_idx >= size_a + size_b)
        {
            *crossing_x = size_b;
            *crossing_y = size_a;
            return;
        }

        // The 0th coord is x, and 1st is y.
        int k_x = 0, k_y = 0, p_x = 0, p_y = 0, q_x = 0, q_y = 0;   // k: bottom left of the diagnal,
                                                                    // p: top right of the diagnal,
                                                                    // q: mid point of the diagnal
        if (diag_idx > size_a)
        {
            k_x = diag_idx - size_a;
            k_y = size_a;
        }
        else
        {
            k_x = 0;
            k_y = diag_idx;
        }
        if (diag_idx > size_b)
        {
            p_x = size_b;
            p_y = diag_idx - size_b;
        }
        else
        {
            p_x = diag_idx;
            p_y = 0;
        }
        while (p_x>=k_x && p_y<=k_y)
        {
            int offset = (k_y - p_y) / 2;
            q_x = k_x + offset;
            q_y = k_y - offset;
            if (q_y == size_a || q_x == 0 || arr_a[q_y] > arr_b[q_x-1])
            {
                if (q_y == 0 || q_x == size_b || arr_a[q_y-1] <= arr_b[q_x])
                {
                    if (q_y < size_a && (q_x == size_b || arr_a[q_y] <= arr_b[q_x]))
                    {
                        arr_tar[diag_idx] = arr_a[q_y];
                        *a_or_b = true;
                    }
                    else
                    {
                        arr_tar[diag_idx] = arr_b[q_x];
                        *a_or_b = false;
                    }
                    // printf("(%d, %d)\n", q_x, q_y);
                    *crossing_x = q_x;
                    *crossing_y = q_y;
                    break;
                }
                else
                {
                    k_x = q_x + 1;
                    k_y = q_y - 1;
                }
            }
            else
            {
                p_x = q_x - 1;
                p_y = q_y + 1;
            }
        }
    }

    template <typename T>
    __device__ void _one_diag(T* arr_tar, T* arr_a, T* arr_b, int diag_idx, int tar_idx, size_t size_a, size_t size_b)
    {
        if (diag_idx >= size_a + size_b){return;}

        // The 0th coord is x, and 1st is y.
        int k_x = 0, k_y = 0, p_x = 0, p_y = 0, q_x = 0, q_y = 0;   // k: bottom left of the diagnal,
                                                                    // p: top right of the diagnal,
                                                                    // q: mid point of the diagnal
        if (diag_idx > size_a)
        {
            k_x = diag_idx - size_a;
            k_y = size_a;
        }
        else
        {
            k_x = 0;
            k_y = diag_idx;
        }
        if (diag_idx > size_b)
        {
            p_x = size_b;
            p_y = diag_idx - size_b;
        }
        else
        {
            p_x = diag_idx;
            p_y = 0;
        }
        while (p_x>=k_x && p_y<=k_y)
        {
            int offset = (k_y - p_y) / 2;
            q_x = k_x + offset;
            q_y = k_y - offset;
            if (q_y == size_a || q_x == 0 || arr_a[q_y] > arr_b[q_x-1])
            {
                if (q_y == 0 || q_x == size_b || arr_a[q_y-1] <= arr_b[q_x])
                {
                    if (q_y < size_a && (q_x == size_b || arr_a[q_y] <= arr_b[q_x]))
                    {
                        arr_tar[tar_idx] = arr_a[q_y];
                    }
                    else
                    {
                        arr_tar[tar_idx] = arr_b[q_x];
                    }
                    break;
                }
                else
                {
                    k_x = q_x + 1;
                    k_y = q_y - 1;
                }
            }
            else
            {
                p_x = q_x - 1;
                p_y = q_y + 1;
            }
        }
    }

    template <typename T>
    __global__ void merge_small_idx_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b)
    {
        int crossing_x = 0;
        int crossing_y = 0;
        int diag_idx = threadIdx.x;
        _unit_merge(arr_tar, arr_a, arr_b, diag_idx, diag_idx, size_a, size_b,
                    &crossing_x, &crossing_y);
        // _one_diag_idx(arr_tar, arr_a, arr_b, diag_idx, size_a, size_b, &crossing_x, &crossing_y);
        // printf("crossing_idx: (%d, %d)\n", crossing_x, crossing_y);
    }


    template <typename T>
    __global__ void merge_big_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b)
    {
        // partition
        // printf("%d %d %d\n", gridDim.x * blockDim.x, size_a + size_b, gridDim.x * blockDim.x < size_a + size_b);
        if (gridDim.x * (blockDim.x + 2) < size_a + size_b || blockDim.x < 2)
        {
            return;
        }  //raise error;
        
        __shared__ int crossing_x[2];
        __shared__ int crossing_y[2];
        __shared__ bool a_or_b[2];
        __shared__ int a_start_idx;
        __shared__ int b_start_idx;
        __shared__ int a_len;
        __shared__ int b_len;
        if (threadIdx.x <= 1)
        {
            auto idx_ = (blockDim.x + 2) * (blockIdx.x + threadIdx.x) - threadIdx.x;
            // _one_diag_idx(arr_tar, arr_a, arr_b, (blockDim.x + 1) * (blockIdx.x + threadIdx.x), size_a, size_b, &(crossing_x[threadIdx.x]), &(crossing_y[threadIdx.x]), &(a_or_b[threadIdx.x]));
            _unit_merge(arr_tar, arr_a, arr_b, idx_, idx_, size_a, size_b,
                        crossing_x+threadIdx.x, crossing_y+threadIdx.x, a_or_b+threadIdx.x);
            // _one_diag_idx(arr_tar, arr_a, arr_b, (blockDim.x + 2) * (blockIdx.x + threadIdx.x) - threadIdx.x, size_a, size_b, 
                            // &(crossing_x[threadIdx.x]), &(crossing_y[threadIdx.x]), &(a_or_b[threadIdx.x]));
        }
        // printf("block %d: from (%d, %d) to (%d, %d)\n", blockIdx.x, crossing_x[0], crossing_y[0], crossing_x[1], crossing_y[1]);
        __syncthreads();
        // merge
        if (threadIdx.x == 0)
        {
            a_start_idx = crossing_y[0] + a_or_b[0];
            b_start_idx = crossing_x[0] + 1 - a_or_b[0];
            a_len = crossing_y[1] - crossing_y[0] - a_or_b[0];
            b_len = crossing_x[1] - crossing_x[0] - 1 + a_or_b[0];
        }
        __syncthreads();
        int tar_idx = blockIdx.x * (blockDim.x + 2) + threadIdx.x + 1;
        // printf("block %d, thread %d, diag_idx=%d\n", blockIdx.x, threadIdx.x, diag_idx);
        // printf("block %d, thread %d, astart %d, bstart %d, alen %d, blen %d, tar %d\n", blockIdx.x, threadIdx.x, a_start_idx, b_start_idx, a_len, b_len, tar_idx);
        auto idx_diag = threadIdx.x;
        _unit_merge(arr_tar, arr_a+a_start_idx, arr_b+b_start_idx, idx_diag, tar_idx, a_len, b_len);
        // _one_diag(arr_tar, &(arr_a[a_start_idx]), &(arr_b[b_start_idx]), threadIdx.x, tar_idx, a_len, b_len);
    }

}