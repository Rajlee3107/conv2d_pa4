NVCC := nvcc
CFLAGS := -O2 --std=c++11
EXTRA_NVCCFLAGS := --cudart=shared

all: conv2dV1 conv2dV2

conv2dV1: conv2dV1.cu
	$(NVCC) $(CFLAGS) conv2dV1.cu -o conv2dV1

conv2dV2: conv2dV2.cu
	$(NVCC) $(CFLAGS) conv2dV2.cu -o conv2dV2

clean:
	rm -f conv2dV1 conv2dV2
