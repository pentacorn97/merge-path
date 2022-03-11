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
    std::uniform_int_distribution<int> gen(0, 1000); // 100000
    // Test the int type.
    constexpr size_t size_a = 48; // 30000
    constexpr size_t size_b = 36; // 35600
    constexpr size_t n_blocks = 12; // 64
    constexpr size_t n_threads = 6; // 1024
    constexpr size_t size_ttl = size_a + size_b;
    int arr_a[size_a] = {}, arr_b[size_b] = {}, arr_m[size_ttl] = {};
    // int crossing_x[n_blocks+1] {0}, crossing_y[n_blocks+1] {0}, a_or_b[n_blocks] {false};
    // crossing_x[n_blocks] = size_b;
    // crossing_y[n_blocks] = size_a;

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

    int *p_a = nullptr, *p_b = nullptr, *p_m = nullptr, *p_crsx, *p_crsy, *p_ab;
    cudaMalloc(&p_a, size_a*sizeof(int));
    cudaMalloc(&p_b, size_b*sizeof(int));
    cudaMalloc(&p_m, size_ttl*sizeof(int));
    cudaMalloc(&p_crsx, (n_blocks+1)*sizeof(int));
    cudaMalloc(&p_crsy, (n_blocks+1)*sizeof(int));
    cudaMalloc(&p_ab, (n_blocks+1)*sizeof(bool));
    cudaMemcpy(p_a, arr_a, size_a*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_b, arr_b, size_b*sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(p_crsx, crossing_x, (n_blocks+1)*sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(p_crsy, crossing_y, (n_blocks+1)*sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Merge_big (a,b)" << std::endl;
    std::fill(arr_m, arr_m+size_ttl, 0);
    cudaMemcpy(p_crsx+n_blocks, &size_b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_crsy+n_blocks, &size_a, sizeof(int), cudaMemcpyHostToDevice);
    merge::partition_k<int><<<n_blocks,n_threads>>>(p_m, p_a, p_b, size_a, size_b, p_crsx, p_crsy, p_ab);
    merge::merge_k<int><<<n_blocks,n_threads>>>(p_m, p_a, p_b, size_a, size_b, p_crsx, p_crsy, p_ab);
    cudaDeviceSynchronize();
    cudaMemcpy(arr_m, p_m, size_ttl*sizeof(int), cudaMemcpyDeviceToHost);
    for(size_t i=0; i<size_ttl; ++i) 
    { std::cout << arr_m[i] << " "; }
    std::cout << (std::is_sorted(arr_m, arr_m+size_ttl) ? "well sorted." : "not sorted.") << std::endl;
    int *crossing_x = (int *)malloc((n_blocks+1)*sizeof(int));
    int *crossing_y = (int *)malloc((n_blocks+1)*sizeof(int));
    int *a_or_b = (int *)malloc((n_blocks+1)*sizeof(bool));
    cudaMemcpy(crossing_x, p_crsx, (n_blocks+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(crossing_y, p_crsy, (n_blocks+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(a_or_b, p_ab, (n_blocks+1)*sizeof(bool), cudaMemcpyDeviceToHost);
    std::cout << "crossing_x: ";
    for(size_t i=0; i<=n_blocks; ++i) 
    { std::cout << crossing_x[i] << " "; }
    std::cout << std::endl;
    std::cout << "crossing_y: ";
    for(size_t i=0; i<=n_blocks; ++i) 
    { std::cout << crossing_y[i] << " "; }
    std::cout << std::endl;
    std::cout << "a_or_b: ";
    for(size_t i=0; i<=n_blocks; ++i) 
    { std::cout << (a_or_b[i] ? "true" : "false") << " "; }
    std::cout << std::endl;

    std::cout << "Merge_big (b,a)" << std::endl;
    std::fill(arr_m, arr_m+size_ttl, 0);
    cudaMemcpy(p_crsx+n_blocks, &size_a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_crsy+n_blocks, &size_b, sizeof(int), cudaMemcpyHostToDevice);
    merge::partition_k<int><<<n_blocks,n_threads>>>(p_m, p_b, p_a, size_b, size_a, p_crsx, p_crsy, p_ab);
    merge::merge_k<int><<<n_blocks,n_threads>>>(p_m, p_b, p_a, size_b, size_a, p_crsx, p_crsy, p_ab);
    cudaDeviceSynchronize();
    cudaMemcpy(arr_m, p_m, size_ttl*sizeof(int), cudaMemcpyDeviceToHost);
    for(size_t i=0; i<size_ttl; ++i) 
    { std::cout << arr_m[i] << " "; }
    std::cout << (std::is_sorted(arr_m, arr_m+size_ttl) ? "well sorted." : "not sorted.") << std::endl;
    cudaMemcpy(crossing_x, p_crsx, (n_blocks+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(crossing_y, p_crsy, (n_blocks+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(a_or_b, p_ab, (n_blocks+1)*sizeof(bool), cudaMemcpyDeviceToHost);
    std::cout << "crossing_x: ";
    for(size_t i=0; i<=n_blocks; ++i) 
    { std::cout << crossing_x[i] << " "; }
    std::cout << std::endl;
    std::cout << "crossing_y: ";
    for(size_t i=0; i<=n_blocks; ++i) 
    { std::cout << crossing_y[i] << " "; }
    std::cout << std::endl;
    std::cout << "a_or_b: ";
    for(size_t i=0; i<=n_blocks; ++i) 
    { std::cout << (a_or_b[i] ? "true" : "false") << " "; }
    std::cout << std::endl;
    return 0;
}