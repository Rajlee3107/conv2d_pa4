#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>

void readInput(const char* fileName, float*& h_input, int& H, int& W) {
    std::ifstream inFile(fileName);
    if (!inFile) {
        std::cerr << "Error: Could not open input file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    inFile >> H >> W;
    int size = H * W;

    // Allocate host memory
    h_input = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < H * W; i++) {
        if (!(inFile >> h_input[i])) {
            std::cerr << "Error: Invalid file format. Expected " << H * W << " elements." << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

void readFilter(const char* fileName, float*& h_filter, int& R) {
    std::ifstream inFile(fileName);
    if (!inFile) {
        std::cerr << "Error: Could not open filter file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    inFile >> R;
    int size = R * R;

    // Allocate host memory
    h_filter = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < R * R; i++) {
        if (!(inFile >> h_filter[i])) {
            std::cerr << "Error: Invalid file format. Expected " << R * R << " elements." << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

void writeOutput(float* h_output, int H, int W) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            std::cout << std::fixed << std::setprecision(3) << h_output[i * W + j] << std::endl;
        }
    }
}

// ------- CUDA Kernel for 2D Convolution -------
__global__ void conv2d_kernel(float* input, float* filter, float* output, int H, int W, int R) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        float sum = 0.0f;
        int pad = R / 2;

        for (int i = -pad; i <= pad; i++) {
            for (int j = -pad; j <= pad; j++) {
                int r = row + i;
                int c = col + j;
                if (r >= 0 && r < H && c >= 0 && c < W) {
                    int filter_row = i + pad;
                    int filter_col = j + pad;
                    sum += input[r * W + c] * filter[filter_row * R + filter_col];
                }
            }
        }
        output[row * W + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "ERROR: Number of arguments < 3" << std::endl;
        return -1;
    }

    // CUDA error check
    cudaError_t err = cudaSuccess;

    // ---- Read Inputs ---- //
    int H, W, R;
    float *h_input, *h_filter, *h_output;
    
    readInput(argv[1], h_input, H, W);
    readFilter(argv[2], h_filter, R);

    // Allocate device memory
    float *d_input, *d_filter, *d_output;
    cudaMalloc((void**)&d_input, H * W * sizeof(float));
    cudaMalloc((void**)&d_filter, R * R * sizeof(float));
    cudaMalloc((void**)&d_output, H * W * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, h_input, H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, R * R * sizeof(float), cudaMemcpyHostToDevice);

    // ---- CUDA Event Timing ---- //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- Define Grid and Block Size ---- //
    dim3 blockDim(32, 32);
    dim3 gridDim((W + 31) / 32, (H + 31) / 32);

    // ---- Record Time and Launch Kernel ---- //
    cudaEventRecord(start);
    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, H, W, R);
    cudaEventRecord(stop);

    // Wait for GPU to finish execution
    cudaDeviceSynchronize();

    // Calculate Execution Time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print Execution Time
    //printf("Kernel Execution Time: %.6f ms\n", milliseconds);

    // ---- Copy results back to host ---- //
    h_output = (float*)malloc(H * W * sizeof(float));
    cudaMemcpy(h_output, d_output, H * W * sizeof(float), cudaMemcpyDeviceToHost);

    // Print Output
    writeOutput(h_output, H, W);

    // ---- Free Memory ---- //
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    free(h_input);
    free(h_filter);
    free(h_output);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

