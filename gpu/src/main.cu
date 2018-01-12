#include <iostream>
#include <algorithm>
#include "cuda_util.h"

//fft kernel
__global__ void fftOvgu(float* data) {

}

//program entry point
int main(int /*argc*/, char** /*argv*/) {

  int n = 8;
  //generate input data
  float* data = (float*) malloc(sizeof(float)*n);
  for(int i = 0; i < n; i++) {
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

  //TODO:determine thread layout


  //TODO:run kernel


  //copy result back
  checkErrorsCuda( cudaMemcpy( data, data_device, sizeof(float) * n, cudaMemcpyDeviceToHost));

  //clean memory
  checkErrorsCuda( cudaFree( data_device));
  free(data);
}
