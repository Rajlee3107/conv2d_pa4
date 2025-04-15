void readInput(const char* fileName, float*& h_input, int& H, int& W, int& N) {
    std::ifstream inFile(fileName);
    if (!inFile) {
        std::cerr << "Error: Could not open input file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    inFile >> H >> W >> N;
    h_input = (float*)malloc(N * H * W * sizeof(float));

    for (int n = 0; n < N; ++n)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                inFile >> h_input[n * H * W + i * W + j];
}

void readFilter(const char* fileName, float*& h_filter, int& K, int& R) {
    std::ifstream inFile(fileName);
    if (!inFile) {
        std::cerr << "Error: Could not open filter file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    inFile >> R;
    inFile >> K;
    h_filter = (float*)malloc(K * R * R * sizeof(float));

    for (int k = 0; k < K; ++k)
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < R; ++j)
                inFile >> h_filter[k * R * R + i * R + j];
}

void writeOutput(float* h_output, int H, int W, int N, int K) {
    for (int nk = 0; nk < N * K; ++nk) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                std::cout << std::fixed << std::setprecision(3) << h_output[nk * H * W + i * W + j];
                if (j != W - 1) std::cout << " ";
            }
            std::cout << std::endl;
        }
    }
}

__global__ void conv2d_shared_kernel(float* input, float* filter, float* output, int H, int W, int N, int K, int R) {
    extern __shared__ float sh_input[];
    __shared__ float sh_filter[MAX_FILTER_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    int nk = blockIdx.z;
    int n = nk / K;
    int k = nk % K;
    int pad = R / 2;

    int sh_W = BLOCK_SIZE + 2 * pad;
    int sh_H = BLOCK_SIZE + 2 * pad;

    int local_r = ty + pad;
    int local_c = tx + pad;

    // Load filter into shared memory (only once per block)
    if (tx == 0 && ty == 0) {
        for (int i = 0; i < R * R; ++i) {
            sh_filter[i] = filter[k * R * R + i];
        }
    }

    // Load shared memory tile
    for (int i = ty; i < sh_H; i += blockDim.y) {
        for (int j = tx; j < sh_W; j += blockDim.x) {
            int global_r = blockIdx.y * BLOCK_SIZE + i - pad;
            int global_c = blockIdx.x * BLOCK_SIZE + j - pad;

            if (global_r >= 0 && global_r < H && global_c >= 0 && global_c < W)
                sh_input[i * sh_W + j] = input[n * H * W + global_r * W + global_c];
            else
                sh_input[i * sh_W + j] = 0.0f;
        }
    }

    __syncthreads();

    if (row < H && col < W) {
        float sum = 0.0f;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < R; ++j) {
                int sh_idx = (local_r + i - pad) * sh_W + (local_c + j - pad);
                int f_idx = i * R + j;
                sum += sh_input[sh_idx] * sh_filter[f_idx];
            }
        }
        output[nk * H * W + row * W + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "ERROR: Number of arguments < 3" << std::endl;
        return -1;
    }

    int H, W, N, K, R;
    float *h_input, *h_filter, *h_output;
    readInput(argv[1], h_input, H, W, N);
    readFilter(argv[2], h_filter, K, R);

    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, N * H * W * sizeof(float));
    cudaMalloc(&d_filter, K * R * R * sizeof(float));
    cudaMalloc(&d_output, N * K * H * W * sizeof(float));

    cudaMemcpy(d_input, h_input, N * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, K * R * R * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE, N * K);
    int sharedMemSize = (BLOCK_SIZE + 2 * (R / 2)) * (BLOCK_SIZE + 2 * (R / 2)) * sizeof(float);

    conv2d_shared_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_filter, d_output, H, W, N, K, R);
    cudaDeviceSynchronize();

    h_output = (float*)malloc(N * K * H * W * sizeof(float));
    cudaMemcpy(h_output, d_output, N * K * H * W * sizeof(float), cudaMemcpyDeviceToHost);

    writeOutput(h_output, H, W, N, K);

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    free(h_input);
    free(h_filter);
    free(h_output);

    return 0;
}


