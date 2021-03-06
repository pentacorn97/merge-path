#include <random>
#include <iostream>
#include "../inc/alg.cuh"

int main(int argc, char* argv[])
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_spent = 0;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> gen(0, 1000);

    // Test the int type.
    constexpr size_t size_a = 512;
    constexpr size_t size_b = 512;
    constexpr size_t size_ttl = size_a + size_b;
    int arr_a[size_a] = {}, arr_b[size_b] = {}, arr_m[size_ttl] = {};

    for(size_t i=0; i<size_a; ++i) { arr_a[i] = gen(generator); }
    for(size_t i=0; i<size_b; ++i) { arr_b[i] = gen(generator); }
    std::sort(arr_a, arr_a+size_a);
    std::sort(arr_b, arr_b+size_b);

    // std::cout << "a: ";
    // for(size_t i=0; i<size_a; ++i) { std::cout << arr_a[i] << " "; }
    // std::cout << std::endl;
    // std::cout << "b: ";
    // for(size_t i=0; i<size_b; ++i) { std::cout << arr_b[i] << " "; }
    // std::cout << std::endl;

    int *p_a = nullptr, *p_b = nullptr, *p_m = nullptr;
    cudaMalloc(&p_a, size_a*sizeof(int));
    cudaMalloc(&p_b, size_b*sizeof(int));
    cudaMalloc(&p_m, size_ttl*sizeof(int));
    cudaMemcpy(p_a, arr_a, size_a*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_b, arr_b, size_b*sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Merge (a, b)" << std::endl;
    std::fill(arr_m, arr_m+size_ttl, 0);

    cudaEventRecord(start, 0);
    for (auto i=0; i<1000; ++i)
    { merge::merge_small_k<int><<<1, 1024, 1024*sizeof(int)>>>(p_m, p_a, p_b, size_a, size_b); }
    cudaEventRecord(stop, 0);

    cudaMemcpy(arr_m, p_m, size_ttl*sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "m: ";
    // for(size_t i=0; i<size_ttl; ++i) 
    // { std::cout << arr_m[i] << " "; }
    cudaEventElapsedTime(&time_spent, start, stop);
    std::cout << (std::is_sorted(arr_m, arr_m+size_ttl) ? "well sorted." : "not sorted.");
    std::cout << "Time used : " << time_spent << "ms." << std::endl;

    // Test b first.
    std::cout << "Merge (b, a)" << std::endl;
    cudaEventRecord(start, 0);
    for (auto i=0; i<1000; ++i)
    { merge::merge_small_k<int><<<1, 1024, 1024*sizeof(int)>>>(p_m, p_b, p_a, size_b, size_a); }
    cudaEventRecord(stop, 0);

    cudaMemcpy(arr_m, p_m, size_ttl*sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "m: ";
    // for(size_t i=0; i<size_ttl; ++i) 
    // { std::cout << arr_m[i] << " "; }
    cudaEventElapsedTime(&time_spent, start, stop);
    std::cout << (std::is_sorted(arr_m, arr_m+size_ttl) ? "well sorted." : "not sorted.");
    std::cout << "Time used : " << time_spent << "ms." << std::endl;

    return 0;
}