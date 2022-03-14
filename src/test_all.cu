#include <random>
#include <iostream>
#include <fstream>
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
    constexpr size_t MIN_SIZE = 1024;
    constexpr size_t MAX_SIZE = 1024*1024;
    constexpr size_t MAX_SIZE_A = 1024*512;
    constexpr size_t MAX_SIZE_B = 1024*512;
    int *arr_a = new int[MAX_SIZE_A];
    int *arr_b = new int[MAX_SIZE_B];
    int *arr_m = new int[MAX_SIZE];
    // int arr_a[MAX_SIZE_A] = {}, arr_b[MAX_SIZE_B] = {}, arr_m[MAX_SIZE] = {};

    for(size_t i=0; i<MAX_SIZE_A; ++i) { arr_a[i] = gen(generator); }
    for(size_t i=0; i<MAX_SIZE_B; ++i) { arr_b[i] = gen(generator); }
    std::sort(arr_a, arr_a+MAX_SIZE_A);
    std::sort(arr_b, arr_b+MAX_SIZE_B);

    int *p_a = nullptr, *p_b = nullptr, *p_m = nullptr;
    // int *p_a = nullptr, *p_b = nullptr, *p_m = nullptr, *p_crsx, *p_crsy, *p_ab;
    // cudaMalloc(&p_a, MAX_SIZE_A*sizeof(int));
    // cudaMalloc(&p_b, MAX_SIZE_B*sizeof(int));
    // cudaMalloc(&p_m, MAX_SIZE*sizeof(int));
    cudaMemcpy(p_a, arr_a, MAX_SIZE_A*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_b, arr_b, MAX_SIZE_B*sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "GPU merge-small, d=1024" << std::endl;
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

    std::cout << "CPU-merge, d=1024" << std::endl;
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
    std::vector<double> vec_t_gpu, vec_t_mt_gpu, vec_t_cpu;
    std::vector<int> vec_d;
    std::cout << "GPU merge-large single-thread" << std::endl;
    for(size_t _size=2; _size<=MAX_SIZE_A; _size*=2 )
    {
        std::fill(arr_m, arr_m+MAX_SIZE, 0);
        size_t _size_m = _size * 2;
        vec_d.push_back(_size_m);
        t_start = std::chrono::high_resolution_clock::now();
        for (size_t i=0; i<REPEAT_TIMES; ++i)
        { merge::merge_big_k<int><<<32768, 1024>>>(p_m, p_a, p_b, _size, _size); }
        t_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> dur_us = (t_stop - t_start);
        cudaMemcpy(arr_m, p_m, _size_m*sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "\t d=" << _size_m << "\t\t";
        std::cout << (std::is_sorted(arr_m, arr_m+_size_m) ? "well sorted." : "not sorted.");
        std::cout << " Time used : " << dur_us.count() << "us." << std::endl;
        vec_t_gpu.push_back(dur_us.count());
    }

    std::cout << "GPU merge-large multi-thread" << std::endl;
    for(size_t _size=2; _size<=MAX_SIZE_A; _size*=2 )
    {
        std::fill(arr_m, arr_m+MAX_SIZE, 0);
        size_t _size_m = _size * 2;
        int n_threads = _size_m > 1024 ? 1024: 64;
        // int n_threads = 64;
        int n_blocks = _size_m / (n_threads + 2);
        if (_size_m % (n_threads + 2) != 0)
            n_blocks++;
        // cudaMalloc(&p_crsx, (n_blocks+1)*sizeof(int));
        // cudaMalloc(&p_crsy, (n_blocks+1)*sizeof(int));
        // cudaMalloc(&p_ab, (n_blocks+1)*sizeof(bool));
        t_start = std::chrono::high_resolution_clock::now();
        for (size_t i=0; i<REPEAT_TIMES; ++i)
        {
            merge::merge_big_2_k<int><<<n_blocks, n_threads>>>(p_m, p_a, p_b, _size, _size);
        }
        t_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> dur_us = (t_stop - t_start);
        cudaMemcpy(arr_m, p_m, _size_m*sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "\t d=" << _size_m << "\t\t";
        std::cout << (std::is_sorted(arr_m, arr_m+_size_m) ? "well sorted." : "not sorted.");
        std::cout << " Time used : " << dur_us.count() << "us." << std::endl;
        vec_t_mt_gpu.push_back(dur_us.count());
    }

    std::cout << "CPU merge-large" << std::endl;
    for(size_t _size=2; _size<=MAX_SIZE_A; _size*=2 )
    {
        std::fill(arr_m, arr_m+size_ttl, 0);
        size_t _size_m = _size * 2;
        t_start = std::chrono::high_resolution_clock::now();
        for (size_t i=0; i<REPEAT_TIMES; ++i)
        { merge::cpu_merge(arr_m, arr_a, arr_b, _size, _size); }
        t_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> dur_us = (t_stop - t_start);
        cudaMemcpy(arr_m, p_m, _size_m*sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "\t d=" << _size_m << "\t\t";
        std::cout << (std::is_sorted(arr_m, arr_m+_size_m) ? "well sorted." : "not sorted.");
        std::cout << " Time used : " << dur_us.count() << "us." << std::endl;
        vec_t_cpu.push_back(dur_us.count());
    }

    if(argc >= 2)
    {
        std::ofstream result(argv[1]);
        result << "d, GPU, CPU \n";
        for (int i=0; i<vec_d.size(); ++i)
        {
            result << vec_d[i] << ", " << vec_t_gpu[i] << "," << vec_t_cpu[i] << "\n";
        }
        result.close();
    }

    cudaFree(p_a);
    cudaFree(p_b);
    cudaFree(p_m);
    delete[] arr_a;
    delete[] arr_b;
    delete[] arr_m;
}