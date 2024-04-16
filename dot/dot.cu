#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <cstdlib>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// CUDA Kernel to compute dot product
__global__ void dotProduct(double *a, double *b, double *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ double temp[256];

    temp[threadIdx.x] = 0;

    for (int i = index; i < n; i += stride) {
        temp[threadIdx.x] += a[i] * b[i];
    }

    __syncthreads();

    // Reduction in shared memory
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            temp[threadIdx.x] += temp[threadIdx.x + i];
        }
        __syncthreads();
    }

    // Write the final sum to global memory
    if (threadIdx.x == 0) {
        atomicAdd(result, temp[0]);
    }
}

// CPU function to compute dot product
double dotProductCPU(std::vector<double>& a, std::vector<double>& b) {
    double result = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <numBlocks> <threadsPerBlock>" << std::endl;
        return 1;
    }

    int numBlocks = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);

    int n = pow(2, 18);
    std::vector<double> a(n); // Initialize vector 'a' with random values
    std::vector<double> b(n); // Initialize vector 'b' with random values

    // Fill vectors with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1.0, 10.0); // Generate random values between 1 and 10
    for (int i = 0; i < n; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    double *d_a, *d_b, *d_result;
    double result = 0.0;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&d_a, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_a!" << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_b, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_b!" << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_result, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_result!" << std::endl;
        return 1;
    }

    // Copy input data to device memory
    cudaMemcpy(d_a, a.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    // Benchmark CUDA implementation
    auto start = std::chrono::high_resolution_clock::now();
    dotProduct<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_result, n);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching dotProduct kernel!" << std::endl;
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Num Blocks: " << numBlocks << ", Threads Per Block: " << threadsPerBlock << ", CUDA time: " << duration.count() << " seconds" << std::endl;

    // Copy result back to host and print
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "CUDA dot product: " << result << std::endl;

    // Benchmark CPU implementation
    start = std::chrono::high_resolution_clock::now();
    result = dotProductCPU(a, b);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CPU time: " << duration.count() << " seconds" << std::endl;
    std::cout << "CPU dot product: " << result << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}