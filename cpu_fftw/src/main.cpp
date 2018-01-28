#include <fftw3.h>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#define NUM_DATAPOINTS 2048
#define REPETITIONS 1024

#define REAL 0
#define IMAG 1

typedef std::chrono::time_point<std::chrono::high_resolution_clock> tpoint;

int main() {
  fftw_complex in[NUM_DATAPOINTS];
  fftw_complex out[NUM_DATAPOINTS];

  fftw_plan p = fftw_plan_dft_1d(NUM_DATAPOINTS, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  
  for (int i = 0; i < NUM_DATAPOINTS; i++) {
    in[i][REAL] = i+1;
    in[i][IMAG] = 0;
  }

  tpoint t_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < REPETITIONS; i++) {
    fftw_execute(p);
  }

  tpoint t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  std::cout << elapsed_time << std::endl;

}
