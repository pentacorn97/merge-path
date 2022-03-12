#include <random>
#include <iostream>
#include <chrono>
#include "../inc/alg.cuh"
#include "../inc/alg_big.cuh"

int main(int argc, char* argv[])
{
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float time_spent = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_stop = std::chrono::high_resolution_clock::now();
    size_t REPEAT_TIMES = 10000;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> gen(0, 1000000);

    ////////////////////////////////////
    // Test merge-small vs single thread.
    std::cout << "Merge small" << std::endl;
    constexpr size_t size_a = 512;
    constexpr size_t size_b = 512;
    constexpr size_t size_ttl = size_a + size_b;
    int arr_a[size_a] = {}, arr_b[size_b] = {}, arr_m[size_ttl] = {};

    for(size_t i=0; i<size_a; ++i) { arr_a[i] = gen(generator); }
    for(size_t i=0; i<size_b; ++i) { arr_b[i] = gen(generator); }
    std::sort(arr_a, arr_a+size_a);
    std::sort(arr_b, arr_b+size_b);

    int *p_a = nullptr, *p_b = nullptr, *p_m = nullptr;
    cudaMalloc(&p_a, size_a*sizeof(int));
    cudaMalloc(&p_b, size_b*sizeof(int));
    cudaMalloc(&p_m, size_ttl*sizeof(int));
    cudaMemcpy(p_a, arr_a, size_a*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_b, arr_b, size_b*sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "GPU merge-small" << std::endl;
    // cudaEventRecord(start, 0);
    t_start = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i<REPEAT_TIMES; ++i)
    { merge::merge_small_k<int><<<1, 1024, 1024*sizeof(int)>>>(p_m, p_a, p_b, size_a, size_b); }
    // cudaEventRecord(stop, 0);
    // cudaEventElapsedTime(&time_spent, start, stop);
    t_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> dur_us = (t_stop - t_start);
    cudaMemcpy(arr_m, p_m, size_ttl*sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "\t" << (std::is_sorted(arr_m, arr_m+size_ttl) ? "well sorted." : "not sorted.");
    std::cout << " Time used : " << dur_us.count() << "us." << std::endl;

    std::cout << "CPU merge" << std::endl;
    std::fill(arr_m, arr_m+size_ttl, 0);
    t_start = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i<REPEAT_TIMES; ++i)
    { merge::cpu_merge(arr_m, arr_a, arr_b, size_a, size_b); }
    t_stop = std::chrono::high_resolution_clock::now();
    dur_us = (t_stop - t_start);
    std::cout << "\t" << (std::is_sorted(arr_m, arr_m+size_ttl) ? "well sorted." : "not sorted.");
    std::cout << " Time used : " << dur_us.count() << "us." << std::endl;

    /////////////////////////////////////////////
    // Test the time of execution of merge-large.
    std::cout << "GPU merge-large" << std::endl;
    for(size_t _size=2; _size<=512; _size*=2 )
    {
        std::fill(arr_m, arr_m+size_ttl, 0);
        size_t _size_m = _size * 2;
        t_start = std::chrono::high_resolution_clock::now();
        for (size_t i=0; i<REPEAT_TIMES; ++i)
        { merge::merge_big_k<int><<<8, 128>>>(p_m, p_a, p_b, size_a, size_b); }
        t_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> dur_us = (t_stop - t_start);
        cudaMemcpy(arr_m, p_m, _size_m*sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "\t d=" << _size_m << "\t";
        std::cout << (std::is_sorted(arr_m, arr_m+_size_m) ? "well sorted." : "not sorted.");
        std::cout << " Time used : " << dur_us.count() << "us." << std::endl;
    }
    std::cout << "CPU merge" << std::endl;
    for(size_t _size=2; _size<=512; _size*=2 )
    {
        std::fill(arr_m, arr_m+size_ttl, 0);
        size_t _size_m = _size * 2;
        t_start = std::chrono::high_resolution_clock::now();
        for (size_t i=0; i<REPEAT_TIMES; ++i)
        { merge::cpu_merge(arr_m, arr_a, arr_b, size_a, size_b); }
        t_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> dur_us = (t_stop - t_start);
        cudaMemcpy(arr_m, p_m, _size_m*sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "\t d=" << _size_m << "\t";
        std::cout << (std::is_sorted(arr_m, arr_m+_size_m) ? "well sorted." : "not sorted.");
        std::cout << " Time used : " << dur_us.count() << "us." << std::endl;
    }

    cudaFree(p_a);
    cudaFree(p_b);
    cudaFree(p_m);
}