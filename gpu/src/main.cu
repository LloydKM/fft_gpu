#include <cmath>
#include <iostream>
#include <algorithm>
#include <thrust/complex.h>
#include "cuda_util.h"

//windows + visualstudio is a truckload of shit
#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

typedef thrust::complex<float> comp;
#define ci comp(0,1)

//fft kernel
template<int DATASIZE>
__global__ void fftOvgu(comp* hdata) {
  //determine thread id
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  //shared memory
  __shared__ comp data[DATASIZE];

  //read data to shared menory by using reversed bitorder
  if (tid == 0) {
    data[0] = hdata[tid];
  } else if (tid == DATASIZE - 1) {
    data[DATASIZE - 1] = hdata[tid];
  } else {
    unsigned int DATA_SIZE = DATASIZE;
    unsigned int new_index = 0;
    int b;
    for (unsigned int i = 0; i < tid; i++) {
      b = DATA_SIZE / 2;
      while (b > 0) {
        if (new_index >= b) {
          new_index -= b;
        } else {
          new_index += b;
          break;
        }
        b /= 2;
      }
    }
    data[new_index] = hdata[tid];
  }

  //TODO: Sync threadblocks
  __syncthreads();

  //going up again and calculate ft
  unsigned int stride = 1;
  unsigned int block_size = 2;
  unsigned int m;
  comp a,b;
  comp quick_math;
  while (stride < DATASIZE) {
    m = tid % stride;
    printf("tid: %d = %d\n");
    quick_math = thrust::exp(comp(-2,0)*ci*comp(M_PI,0)*comp(m,0)/comp(block_size,0));
    if ((tid % block_size) < (block_size/2)) {
      printf("tid %d entered\n", tid);
      a = data[tid];
      b = data[tid+stride]*quick_math;
      data[tid] = a + b;
      data[tid+stride] = a - b;
    }    
    block_size*= 2;
    stride *= 2;
    __syncthreads();
  }

  //copy back shared memory
  hdata[tid] = data[tid];
}

//program entry point
int main(int /*argc*/, char** /*argv*/) {

  const int n = 8;
  //generate input data
  comp* data = (comp*) malloc(sizeof(comp)*n);
  for (int i = 0; i < n; i++) {
    data[i] = comp(i+1,0);
  }

  //check execution environnement
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cout << "Hobo aint got no money for nvidia graphic card!" << std::endl;
    return EXIT_FAILURE;
  }

  //query the device properties
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  printDeviceProps(devProp);

  //set the device
  int device_handle = 0;
  cudaSetDevice(device_handle);

  //init memory aand allocate device memory
  comp* data_device = nullptr;
  checkErrorsCuda( cudaMalloc((void **) &data_device, sizeof(comp) * n));

  //copy device memory
  checkErrorsCuda( cudaMemcpy( (void*) data_device, data, sizeof(comp) * n, cudaMemcpyHostToDevice ));

  //determine thread layout
  const int MAX_THREADS_PER_BLOCK = devProp.maxThreadsPerBlock;
  int num_threads_per_block = std::min(n, MAX_THREADS_PER_BLOCK);
  int num_blocks = n/MAX_THREADS_PER_BLOCK;
  if( 0 != n % MAX_THREADS_PER_BLOCK) {
    num_blocks++;
  }
  std::cout << "num_blocks = " << num_blocks << "num_threads_per_block = " << num_threads_per_block << std::endl;


  //run kernel
  fftOvgu<n> <<< num_blocks, num_threads_per_block >>> (data_device); 
  
  //print result
  for (int i = 0; i < n; i++) {
    std::cout << data[i] << std::endl;
  }

  //copy result back
  checkErrorsCuda( cudaMemcpy( data, data_device, sizeof(comp) * n, cudaMemcpyDeviceToHost));

  //clean memory
  checkErrorsCuda( cudaFree( data_device));
  free(data);
  return EXIT_SUCCESS;
}
