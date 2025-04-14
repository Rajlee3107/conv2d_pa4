#include <iostream>
#include <fstream>
#include <iomanip>
#include <cuda.h>

#define R 3
#define BLOCK_SIZE 16

// ------- Read Input -------
void readInput(const char* filename, float*& input, int& H, int& W, int& N) {
    std::ifstream file(filename);
    file >> H >> W >> N;
    input = new float[N * H * W];
    for (int n = 0; n < N; ++n)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                file >> input[n * H * W + i * W + j];
}

// ------- Read Filter -------
void readFilter(const char* filename, float*& filter, int& K) {
    std::ifstream file(filename);
    file >> K;
    filter = new float[K * R * R];
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < R; ++j)
                file >> filter[k * R * R + i * R + j];
}

// ------- Write Output -------
void writeOutput(float* output, int H, int W, int N, int K) {
    for (int nk = 0; nk < N * K; ++nk) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                std::cout << std::fixed << std::setprecision(3) << output[nk * H * W + i * W + j];
                if (j != W - 1) std::cout << " ";
            }
            std::cout << std::endl;
        }
    }
}

// ------- CUDA Kernel -------
__global__ void conv2d_kernel(float* input, float* filter, float* output, int H, int W, int N, int K) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    int nk = blockIdx.z; // output index
    int n = nk / K;
    int k = nk % K;

    if (row < H && col < W) {
        float sum = 0.0f;
        int pad = R / 2;
        for (int i = -pad; i <= pad; ++i)
            for (int j = -pad; j <= pad; ++j) {
                int r = row + i;
                int c = col + j;
                if (r >= 0 && r < H && c >= 0 && c < W) {
                    int in_idx = n * H * W + r * W + c;
                    int f_idx = k * R * R + (i + pad) * R + (j + pad);
                    sum += input[in_idx] * filter[f_idx];
                }
            }
        output[nk * H * W + row * W + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./conv2dV1 input.txt filter.txt" << std::endl;
        return 1;
    }

    float *input, *filter;
    int H, W, N, K, R;
    readInput(argv[1], input, H, W, N);
    readFilter(argv[2], filter, K);

    float *d_input, *d_filter, *d_output;
    cudaMallocManaged(&d_input, N * H * W * sizeof(float));
    cudaMallocManaged(&d_filter, K * R * R * sizeof(float));
    cudaMallocManaged(&d_output, N * K * H * W * sizeof(float));

    std::copy(input, input + N * H * W, d_input);
    std::copy(filter, filter + K * R * R, d_filter);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE, N * K);
    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, H, W, N, K);
    cudaDeviceSynchronize();

    if (N == 1 && K == 1) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                std::cout << std::fixed << std::setprecision(3) << d_output[i * W + j];
                if (j != W - 1) std::cout << " ";
            }
            std::cout << std::endl;
        }
    } else {
        writeOutput(d_output, H, W, N, K);
    }

    delete[] input;
    delete[] filter;
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    return 0;
}
