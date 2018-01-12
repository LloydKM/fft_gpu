#include <cmath>
#include <iostream>
#include <algorithm>
#include "cuda_util.h"

//fft kernel
__global__ void fftOvgu(float* data) {

}

//program entry point
int main(int /*argc*/, char** /*argv*/) {

  const int n = 8;
  //generate input data
  float* data = (float*) malloc(sizeof(float)*n);
  for (int i = 0; i < n; i++) {
    data[i] = (float) i+1;
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
  float* data_device = nullptr;
  checkErrorsCuda( cudaMalloc((void **) &data_device, sizeof(float) * n));

  //copy device memory
  checkErrorsCuda( cudaMemcpy( (void*) data_device, data, sizeof(float) * n, cudaMemcpyHostToDevice ));

  //determine thread layout
  const int MAX_THREADS_PER_BLOCK = devProp.maxThreadsPerBlock;
  int num_threads_per_block = std::min(n, MAX_THREADS_PER_BLOCK);
  int num_blocks = n/MAX_THREADS_PER_BLOCK;
  if( 0 != n % MAX_THREADS_PER_BLOCK) {
    num_blocks++;
  }
  std::cout << "num_blocks = " << num_blocks << "num_threads_per_block = " << num_threads_per_block << std::endl;


  //run kernel
  fftOvgu <<< num_blocks, num_threads_per_block >>> (data_device); 

  //copy result back
  checkErrorsCuda( cudaMemcpy( data, data_device, sizeof(float) * n, cudaMemcpyDeviceToHost));

  //clean memory
  checkErrorsCuda( cudaFree( data_device));
  free(data);
  return EXIT_SUCCESS;
}
