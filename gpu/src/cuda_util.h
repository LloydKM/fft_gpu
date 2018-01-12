/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : Exercise 1
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: Cuda utility functions
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

// includes, system
#include <iostream>


template< typename T >
void checkError(T result, char const *const func, const char *const file, int const line) {

  if (result)  {
      std::cerr << "CUDA error at " << file << "::" << line << " with error code "
                <<  static_cast<int>(result) << " for " << func << "()." << std::endl;
      cudaDeviceReset();
      exit(EXIT_FAILURE);
  }
}

#define checkErrorsCuda(val) checkError( (val), #val, __FILE__, __LINE__ )


inline void
checkLastCudaErrorFunc(const char *errorMessage, const char *file, const int line){

    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkLastCudaError(msg)  checkLastCudaErrorFunc(msg, __FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Print device properties
////////////////////////////////////////////////////////////////////////////////////////////////////
void
printDeviceProps( const cudaDeviceProp& devProp) {

  printf("Major revision number:         %d\n", devProp.major);
  printf("Minor revision number:         %d\n", devProp.minor);
  printf("Name:                          %s\n", devProp.name);
  printf("Total global memory:           %i\n MB", (int) (devProp.totalGlobalMem / 1048576));
  printf("Total shared memory per block: %i\n", (int) devProp.sharedMemPerBlock);
  printf("Total registers per block:     %d\n", devProp.regsPerBlock);
  printf("Warp size:                     %d\n", devProp.warpSize);
  printf("Maximum memory pitch:          %i\n", (int) devProp.memPitch);
  printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i) {
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
  }
  for (int i = 0; i < 3; ++i) {
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
  }
  printf("Clock rate:                    %d\n", devProp.clockRate);
  printf("Total constant memory:         %i\n", (int) devProp.totalConstMem);
  printf("Texture alignment:             %i\n", (int) devProp.textureAlignment);
  printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
}


#endif // _CUDA_UTIL_H_
