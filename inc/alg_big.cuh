/**
 * @file alg_big.cuh
 * @author  bx.zh, pentacorn97
 * @brief  Declares the algorithms for big arrays
 * @date 2022-03-11
 * 
 * @copyright Copyleft (c) 2022 all wrongs reserved.
 * 
 */
#pragma once
#include <cstdio>
#include <algorithm>
#include <assert.h>

#include "./alg.cuh"

namespace merge
{
    /**
     * @brief The function used to merge big (siza_a+size_b>1024) tables.
     *        This function uses single-thread partition and multi-thread merge
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

    /**
     * @brief The function used to partition big (siza_a+size_b>1024) tables.
     *        This function uses multi-thread partition
     * 
     * @tparam T Type of data.
     * @param arr_tar The target array.
     * @param arr_a The first array. 
     * @param arr_b The second array.
     * @param size_a The size of the first array.
     * @param size_b The size of the second array.
     * @param crossing_x The array used to store x idices of intersection points.
     * @param crossing_y The array used to store y idices of intersection points.
     * @param a_or_b true: arr_a value taken at intersection; false: arr_b value taken at intersection.
     */
    template <typename T>
    __global__ void partition_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b, int* crossing_x, int* crossing_y, int* a_or_b);

    /**
     * @brief The function used to merge big (siza_a+size_b>1024) tables, given the partition results.
     *        This function uses multi-thread merge
     * 
     * @tparam T Type of data.
     * @param arr_tar The target array.
     * @param arr_a The first array. 
     * @param arr_b The second array.
     * @param size_a The size of the first array.
     * @param size_b The size of the second array.
     * @param crossing_x The array used to store x idices of intersection points.
     * @param crossing_y The array used to store y idices of intersection points.
     * @param a_or_b true: arr_a value taken at intersection; false: arr_b value taken at intersection.
     */
    template <typename T>
    __global__ void merge_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b, int* crossing_x, int* crossing_y, int* a_or_b);
    
    template <typename T>
    __global__ void merge_big_2_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b);

    template <typename T>
    __global__ void merge_big_3_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b);


    /////////////////////////////////////////////////////////////
    // Implementation
    /////////////////////////////////////////////////////////////
    template <typename T>
    __global__ void merge_big_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b)
    {
        // partition
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
            _unit_merge<T>(arr_tar, arr_a, arr_b, idx_, idx_, size_a, size_b,
                        crossing_x+threadIdx.x, crossing_y+threadIdx.x, a_or_b+threadIdx.x);
        }
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
        auto idx_diag = threadIdx.x;
        _unit_merge<T>(arr_tar, arr_a+a_start_idx, arr_b+b_start_idx, idx_diag, tar_idx, a_len, b_len);
        // _one_diag(arr_tar, &(arr_a[a_start_idx]), &(arr_b[b_start_idx]), threadIdx.x, tar_idx, a_len, b_len);
    }

    template <typename T>
    __global__ void partition_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b, int* crossing_x, int* crossing_y, int* a_or_b)
    {
        if (gridDim.x * (blockDim.x + 1) < size_a + size_b || blockDim.x < 2)
            return; // raise error;
        __shared__ bool stop; // if stop is true, all threads proceed to merge
        __shared__ int idx_partition_diag;
        __shared__ int ll_x;
        __shared__ int ll_y;
        __shared__ int tr_y;
        __shared__ int len_diag;
        __shared__ int warpDim;
        __shared__ int increment;
        if (threadIdx.x == 0)
        {
            stop = false;
            idx_partition_diag = (blockDim.x + 1) * blockIdx.x;
            // find out low-left and top-right coordinates of the diagnal
            if (idx_partition_diag > size_a)
            {
                ll_x = idx_partition_diag - size_a;
                ll_y = size_a;
            }
            else
            {
                ll_x = 0;
                ll_y = idx_partition_diag;
            }
            if (idx_partition_diag > size_b)
            {
                tr_y = idx_partition_diag - size_b;
            }
            else
            {
                tr_y = 0;
            }
            len_diag = ll_y - tr_y + 1;
        }
        if (threadIdx.x == 1)
        {
            warpDim = warpSize > blockDim.x ? blockDim.x : warpSize;
            increment = blockDim.x - int(blockDim.x/warpDim);
        }
        __syncthreads();
        // each thread takes care of one point of the diagnal
        // printf("block %d, thread %d, partion_diag_idx %d, len_diag %d, increment %d\n", blockIdx.x, threadIdx.x, idx_partition_diag, len_diag, increment);
        for (int start = 0; start < len_diag && (!stop); start += increment)
        {
            int warpIdx = threadIdx.x / warpDim;
            // coordinates of the point treated by this thread
            int idx_a = ll_y-start-threadIdx.x+warpIdx;
            int idx_b = ll_x+start+threadIdx.x-warpIdx;
            // printf("block %d, thread %d, warp %d, partion_diag_idx %d, idx_a %d, idx_b %d\n", blockIdx.x, threadIdx.x, warpIdx, idx_partition_diag, idx_a, idx_b);
            // a, b, a_prev, b_prev are the 4 adjacent values of the two array
            // used to check if the current point is the intersection
            T a, b_prev;
            if (idx_a < 0 || idx_b > size_b)
                break;
            if (idx_a >= 0 && idx_a < size_a)
                a = arr_a[idx_a];
            if (idx_b > 0 && idx_b <= size_b)
                b_prev = arr_b[idx_b-1];
            
            T a_prev = __shfl_down_sync(0xffffffff, a, 1, warpDim);
            T b = __shfl_down_sync(0xffffffff, b_prev, 1, warpDim);
            // printf("block %d, thread %d, warp %d, partion_diag_idx %d, idx_a %d, idx_b %d, a %d, b %d, a_prev %d, b_prev %d,\n", blockIdx.x, threadIdx.x, warpIdx, idx_partition_diag, idx_a, idx_b, a, b, a_prev, b_prev);
            
            if ((idx_a == size_a || idx_b == 0 || a > b_prev) && (idx_a == 0 || idx_b == size_b || a_prev <= b))
            {
                stop = true;
                crossing_x[blockIdx.x] = idx_b;
                crossing_y[blockIdx.x] = idx_a;
                if (idx_a < size_a && (idx_b == size_b || a <= b))
                {
                    arr_tar[idx_partition_diag] = a;
                    a_or_b[blockIdx.x] = true;
                }
                else
                {
                    arr_tar[idx_partition_diag] = b;
                    a_or_b[blockIdx.x] = false;
                }
                // printf("NICE! block %d, thread %d, idx_a %d, idx_b %d, a %d, b %d\n", blockIdx.x, threadIdx.x, idx_a, idx_b, a, b);
                // printf("arr_tar[%d]=%d\n", idx_partition_diag, arr_tar[idx_partition_diag]);
            }
        }
    }


    template <typename T>
    __global__ void merge_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b, int* crossing_x, int* crossing_y, int* a_or_b)
    {
        if (gridDim.x * (blockDim.x + 1) < size_a + size_b)
            return;
        __shared__ int idx_a_start;
        __shared__ int idx_b_start;
        __shared__ int a_len;
        __shared__ int b_len;
        if (threadIdx.x == 0)
        {
            idx_a_start = crossing_y[blockIdx.x] + a_or_b[blockIdx.x];
            idx_b_start = crossing_x[blockIdx.x] - a_or_b[blockIdx.x] + 1;
            a_len = crossing_y[blockIdx.x+1] - crossing_y[blockIdx.x] - a_or_b[blockIdx.x];
            b_len = crossing_x[blockIdx.x+1] - crossing_x[blockIdx.x] + a_or_b[blockIdx.x] - 1;
        }
        int idx_tar = blockIdx.x * (blockDim.x + 1) + threadIdx.x + 1;
        int idx_diag = threadIdx.x;
        __syncthreads();
        // printf("block %d, thread %d, a_len %d, b_len %d, idx_astart %d, idx_bstart %d, idx_tar %d, idx_diag %d\n",
        //        blockIdx.x, threadIdx.x, a_len, b_len, idx_a_start, idx_b_start, idx_tar, idx_diag);
        _unit_merge<T>(arr_tar, arr_a+idx_a_start, arr_b+idx_b_start, idx_diag, idx_tar, a_len, b_len);
    }


    template <typename T>
    __global__ void merge_big_2_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b)
    {
        if (gridDim.x * (blockDim.x + 2) < size_a + size_b || blockDim.x < 2)
            return; // raise error;
        __shared__ int crossing_x[2];
        __shared__ int crossing_y[2];
        __shared__ bool a_or_b[2];
        __shared__ int a_start_idx;
        __shared__ int b_start_idx;
        __shared__ int a_len;
        __shared__ int b_len;
        __shared__ int stop; // if stop is 2, all threads proceed to merge
        if (threadIdx.x == 0)
            stop = 0;
        __syncthreads();
        int half_blockDim = blockDim.x / 2;
        int first_or_second_half = threadIdx.x / half_blockDim; // 0 if in the first half_blockDim threads, 1 else
        int idx_partition_diag = (blockDim.x + 2) * (blockIdx.x + first_or_second_half) - first_or_second_half;
        int ll_x;
        int ll_y;
        int tr_y;
        if (idx_partition_diag > size_a)
        {
            ll_x = idx_partition_diag - size_a;
            ll_y = size_a;
        }
        else
        {
            ll_x = 0;
            ll_y = idx_partition_diag;
        }
        if (idx_partition_diag > size_b)
        {
            tr_y = idx_partition_diag - size_b;
        }
        else
        {
            tr_y = 0;
        }
        int len_diag = ll_y - tr_y + 1;
        int warpDim = warpSize > half_blockDim ? half_blockDim : warpSize;
        int increment = half_blockDim - int(half_blockDim/warpDim);
        // each thread takes care of one point of the diagnal
        // printf("block %d, thread %d, idx_partition_diag %d, len_diag %d, increment %d\n", blockIdx.x, threadIdx.x, idx_partition_diag, len_diag, increment);
        for (int start = 0; (start < len_diag) && (stop < 2); start += increment)
        {
            int idx_thread = threadIdx.x - first_or_second_half * half_blockDim;
            int warpIdx = idx_thread / warpDim;
            // coordinates of the point treated by this thread
            int idx_a = ll_y-start-idx_thread+warpIdx;
            int idx_b = ll_x+start+idx_thread-warpIdx;
            // printf("block %d, thread %d, first/second %d, warp %d, idx_partition_diag %d, idx_a %d, idx_b %d\n", blockIdx.x, threadIdx.x, first_or_second_half, warpIdx, idx_partition_diag, idx_a, idx_b);
            // a, b, a_prev, b_prev are the 4 adjacent values of the two arrays
            // used to check if the current point is the intersection
            T a, b_prev;
            if (idx_a < 0 || idx_b > size_b)
                break;
            if (idx_a >= 0 && idx_a < size_a)
                a = arr_a[idx_a];
            if (idx_b > 0 && idx_b <= size_b)
                b_prev = arr_b[idx_b-1];
            
            T a_prev = __shfl_down_sync(0xffffffff, a, 1, warpDim);
            T b = __shfl_down_sync(0xffffffff, b_prev, 1, warpDim);
            // printf("block %d, thread %d, first/second %d, warp %d, idx_partion_diag %d, idx_a %d, idx_b %d, a %d, b %d, a_prev %d, b_prev %d,\n", blockIdx.x, threadIdx.x, first_or_second_half, warpIdx, idx_partition_diag, idx_a, idx_b, a, b, a_prev, b_prev);
            
            if ((idx_a == size_a || idx_b == 0 || a > b_prev) && (idx_a == 0 || idx_b == size_b || a_prev <= b))
            {
                stop++;
                crossing_x[first_or_second_half] = idx_b;
                crossing_y[first_or_second_half] = idx_a;
                if (idx_a < size_a && (idx_b == size_b || a <= b))
                {
                    arr_tar[idx_partition_diag] = a;
                    a_or_b[first_or_second_half] = true;
                }
                else
                {
                    arr_tar[idx_partition_diag] = b;
                    a_or_b[first_or_second_half] = false;
                }
                // printf("NICE! block %d, thread %d, idx_a %d, idx_b %d, a %d, b %d\n", blockIdx.x, threadIdx.x, idx_a, idx_b, a, b);
                // printf("arr_tar[%d]=%d\n", idx_partition_diag, arr_tar[idx_partition_diag]);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            stop++;
            a_start_idx = crossing_y[0] + a_or_b[0];
            b_start_idx = crossing_x[0] + 1 - a_or_b[0];
            a_len = crossing_y[1] - crossing_y[0] - a_or_b[0];
            b_len = crossing_x[1] - crossing_x[0] - 1 + a_or_b[0];
            // printf("stop=2 in block %d, thread %d\n", blockIdx.x, threadIdx.x);
        }
        __syncthreads();
        int idx_tar = blockIdx.x * (blockDim.x + 2) + threadIdx.x + 1;
        auto idx_diag = threadIdx.x;
        _unit_merge<T>(arr_tar, arr_a+a_start_idx, arr_b+b_start_idx, idx_diag, idx_tar, a_len, b_len);
    }


    template <typename T>
    __global__ void merge_big_3_k(T* arr_tar, T* arr_a, T* arr_b, size_t size_a, size_t size_b)
    {
        if (gridDim.x * (blockDim.x + 2) < size_a + size_b || blockDim.x < 2)
            return; // raise error;
        __shared__ int crossing_x[2];
        __shared__ int crossing_y[2];
        __shared__ bool a_or_b[2];
        __shared__ int a_start_idx;
        __shared__ int b_start_idx;
        __shared__ int a_len;
        __shared__ int b_len;
        __shared__ int stop; // if stop is 2, all threads proceed to merge
        if (threadIdx.x == 0)
            stop = 0;
        __syncthreads();
        int half_blockDim = blockDim.x / 2;
        int first_or_second_half = threadIdx.x / half_blockDim; // 0 if in the first half_blockDim threads, 1 else
        int idx_partition_diag = (blockDim.x + 2) * (blockIdx.x + first_or_second_half) - first_or_second_half;
        int ll_x;
        int ll_y;
        int tr_y;
        if (idx_partition_diag > size_a)
        {
            ll_x = idx_partition_diag - size_a;
            ll_y = size_a;
        }
        else
        {
            ll_x = 0;
            ll_y = idx_partition_diag;
        }
        if (idx_partition_diag > size_b)
        {
            tr_y = idx_partition_diag - size_b;
        }
        else
        {
            tr_y = 0;
        }
        int len_diag = ll_y - tr_y + 1;
        int warpDim = warpSize > half_blockDim ? half_blockDim : warpSize;
        int increment = half_blockDim - int(half_blockDim/warpDim);
        // each thread takes care of one point of the diagnal
        // printf("block %d, thread %d, idx_partition_diag %d, len_diag %d, increment %d\n", blockIdx.x, threadIdx.x, idx_partition_diag, len_diag, increment);
        for (int start = 0; (start < len_diag) && (stop < 2); start += increment)
        {
            int idx_thread = threadIdx.x - first_or_second_half * half_blockDim;
            int warpIdx = idx_thread / warpDim;
            // coordinates of the point treated by this thread
            int idx_a = ll_y-start-idx_thread+warpIdx;
            int idx_b = ll_x+start+idx_thread-warpIdx;
            // printf("block %d, thread %d, first/second %d, warp %d, idx_partition_diag %d, idx_a %d, idx_b %d\n", blockIdx.x, threadIdx.x, first_or_second_half, warpIdx, idx_partition_diag, idx_a, idx_b);
            // a, b, a_prev, b_prev are the 4 adjacent values of the two array
            // used to check if the current point is the intersection
            T a, b_prev;
            if (idx_a < 0 || idx_b > size_b)
                break;
            if (idx_a >= 0 && idx_a < size_a)
                a = arr_a[idx_a];
            if (idx_b > 0 && idx_b <= size_b)
                b_prev = arr_b[idx_b-1];
            
            T a_prev = __shfl_down_sync(0xffffffff, a, 1, warpDim);
            T b = __shfl_down_sync(0xffffffff, b_prev, 1, warpDim);
            printf("block %d, thread %d, first/second %d, warp %d, idx_partion_diag %d, idx_a %d, idx_b %d, a %d, b %d, a_prev %d, b_prev %d,\n", blockIdx.x, threadIdx.x, first_or_second_half, warpIdx, idx_partition_diag, idx_a, idx_b, a, b, a_prev, b_prev);
            
            if ((idx_a == size_a || idx_b == 0 || a > b_prev) && (idx_a == 0 || idx_b == size_b || a_prev <= b))
            {
                stop++;
                crossing_x[first_or_second_half] = idx_b;
                crossing_y[first_or_second_half] = idx_a;
                if (idx_a < size_a && (idx_b == size_b || a <= b))
                {
                    arr_tar[idx_partition_diag] = a;
                    a_or_b[first_or_second_half] = true;
                }
                else
                {
                    arr_tar[idx_partition_diag] = b;
                    a_or_b[first_or_second_half] = false;
                }
                // printf("NICE! block %d, thread %d, idx_a %d, idx_b %d, a %d, b %d\n", blockIdx.x, threadIdx.x, idx_a, idx_b, a, b);
                // printf("arr_tar[%d]=%d\n", idx_partition_diag, arr_tar[idx_partition_diag]);
            }
        }


        
        __syncthreads();
        if (threadIdx.x == 0)
        {
            stop++;
            a_start_idx = crossing_y[0] + a_or_b[0];
            b_start_idx = crossing_x[0] + 1 - a_or_b[0];
            a_len = crossing_y[1] - crossing_y[0] - a_or_b[0];
            b_len = crossing_x[1] - crossing_x[0] - 1 + a_or_b[0];
            // printf("stop=2 in block %d, thread %d\n", blockIdx.x, threadIdx.x);
        }
        __syncthreads();
        int idx_tar = blockIdx.x * (blockDim.x + 2) + threadIdx.x + 1;
        auto idx_diag = threadIdx.x;
        _unit_merge<T>(arr_tar, arr_a+a_start_idx, arr_b+b_start_idx, idx_diag, idx_tar, a_len, b_len);
    }
}