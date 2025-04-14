#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>

void readInput(const char* fileName, float*& h_input, int& H, int& W) {
    std::ifstream inFile(fileName);
    if (!inFile) {
        std::cerr << "Error: Could not open Input file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    inFile >> H >> W; // Using the extraction operator
    int size = H * W;
    
    // Allocate Unified Memory for the input matrix
    cudaMallocManaged(&h_input, size * sizeof(float));

    for(int i = 0; i < H * W; i++) {
        if(!(inFile >> h_input[i])) {
            std::cerr << "Error: Invalid file format. Expected " << H * W << " elements." << std::endl;
            exit(EXIT_FAILURE); 
        }
    }
}

void readFilter(const char* fileName, float*& h_filter, int& R) {
    std::ifstream inFile(fileName);
    if(!inFile) {
        std::cerr << "Error: Could not open Filter file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    inFile >> R;
    int size = R * R;

    // Allocate Unified Memory for the filter
    cudaMallocManaged(&h_filter, size * sizeof(float));

    for(int i = 0; i < R * R; i++) {
        if(!(inFile >> h_filter[i])) {
            std::cerr << "Error: Invalid file format. Expected " << R * R << " elements." << std::endl;
            exit(EXIT_FAILURE); 
        }
    }
}

void writeOutput(float* h_output, int H, int W) {
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            std::cout << std::fixed << std::setprecision(3) << h_output[i * W + j] << std::endl;
        }
    }
}

// ------- The CUDA Kernel -------
__global__ void conv2d_kernel(float* input, float* filter, float* output, int H, int W, int R) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        float sum = 0.0f;
        int pad = R / 2;
        
        for(int i = -pad; i <= pad; i++) {
            for(int j = -pad; j <= pad; j++) {
                int r = row + i;
                int c = col + j;
                if(r >= 0 && r < H && c >= 0 && c < W) {
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
    if(argc < 3) {
        std::cerr << "ERROR: number of arguments < 3" << std::endl;
        return -1;
    }

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // ---- Read the inputs from command line ---- //
    int H, W, R;
    float *input, *filter, *output;
    
    readInput(argv[1], input, H, W);
    readFilter(argv[2], filter, R);

    //printf("[2D Convolution of %dx%d image with %dx%d filter]\n", H, W, R, R);

    // ---- Allocate Unified Memory for output ---- //
    cudaMallocManaged(&output, H * W * sizeof(float));

    // ---- Create CUDA events for timing ---- //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- Launch the kernel with timing ---- //
    dim3 blockDim(32, 32);
    dim3 gridDim((W + 31) / 32, (H + 31) / 32);

    //printf("CUDA kernel launch with %d x %d blocks of %d x %d threads\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    cudaEventRecord(start);  // Start timing
    conv2d_kernel<<<gridDim, blockDim>>>(input, filter, output, H, W, R);
    cudaEventRecord(stop);   // Stop timing

    // Wait for GPU to finish before accessing results
    cudaDeviceSynchronize();

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print execution time
    //printf("Kernel Execution Time: %.6f ms\n", milliseconds);

    // Print the output
    writeOutput(output, H, W);
    //printf("Test Completed\n");

    // ---- Clean up the memory ---- //
    cudaFree(input);
    cudaFree(filter);
    cudaFree(output);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Reset the device
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //printf("Done\n");
    return 0;
}

