#include <random>
#include <iostream>
#include "../inc/alg.cuh"
#include "../inc/alg_big.cuh"

__global__ void test_idxs()
{
    printf("threadIdx.x = %d, blockDim.x = %d, blockIdx.x = %d, girdDim.x = %d\n", threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);
}

int main(int argc, char* argv[])
{
    // test_idxs<<<2,3>>>();
    
    std::default_random_engine generator;
    std::uniform_int_distribution<int> gen(0, 1000);
    // Test the int type.
    constexpr size_t size_a = 26666;
    constexpr size_t size_b = 38998;
    constexpr size_t size_ttl = size_a + size_b;
    int arr_a[size_a] = {}, arr_b[size_b] = {}, arr_m[size_ttl] = {};

    for(size_t i=0; i<size_a; ++i) { arr_a[i] = gen(generator); }
    for(size_t i=0; i<size_b; ++i) { arr_b[i] = gen(generator); }
    std::sort(arr_a, arr_a+size_a);
    std::sort(arr_b, arr_b+size_b);

    std::cout << "a:\n";
    for(size_t i=0; i<size_a; ++i) { std::cout << arr_a[i] << " "; }
    std::cout << std::endl;
    std::cout << "b:\n";
    for(size_t i=0; i<size_b; ++i) { std::cout << arr_b[i] << " "; }
    std::cout << std::endl;

    int *p_a = nullptr, *p_b = nullptr, *p_m = nullptr;
    cudaMalloc(&p_a, size_a*sizeof(int));
    cudaMalloc(&p_b, size_b*sizeof(int));
    cudaMalloc(&p_m, size_ttl*sizeof(int));
    cudaMemcpy(p_a, arr_a, size_a*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_b, arr_b, size_b*sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Merge_big (a,b)" << std::endl;
    std::fill(arr_m, arr_m+size_ttl, 0);
    merge::merge_big_k<int><<<64,1024>>>(p_m, p_a, p_b, size_a, size_b);
    cudaDeviceSynchronize();
    cudaMemcpy(arr_m, p_m, size_ttl*sizeof(int), cudaMemcpyDeviceToHost);
    for(size_t i=0; i<size_ttl; ++i) 
    { std::cout << arr_m[i] << " "; }
    std::cout << (std::is_sorted(arr_m, arr_m+size_ttl) ? "well sorted." : "not sorted.") << std::endl;

    std::cout << "Merge_big (b,a)" << std::endl;
    std::fill(arr_m, arr_m+size_ttl, 0);
    merge::merge_big_k<int><<<64,1024>>>(p_m, p_b, p_a, size_b, size_a);
    cudaDeviceSynchronize();
    cudaMemcpy(arr_m, p_m, size_ttl*sizeof(int), cudaMemcpyDeviceToHost);
    for(size_t i=0; i<size_ttl; ++i) 
    { std::cout << arr_m[i] << " "; }
    std::cout << (std::is_sorted(arr_m, arr_m+size_ttl) ? "well sorted." : "not sorted.") << std::endl;
    return 0;
}