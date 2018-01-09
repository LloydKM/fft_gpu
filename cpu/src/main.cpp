#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <complex>
#include <iomanip>

typedef std::complex<double> comp;
#define li comp(0,1)

std::vector<comp> fft(int n, std::vector<comp> freqs);

int main(int argc, char** argv) {
  std::vector<comp> test = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<comp> fftv = fft(test.size(), test);
  std::cout << fftv.size() << std::endl;
  for (auto const& value: fftv) {
    std::cout << value << std::endl;
  }  
}

std::vector<comp> fft(int n, std::vector<comp> freqs) {
  //trivial case - return solution
  if ( n == 1) {
    return freqs;
  }

  //split freq vector in vector with even and vector with odd values
  std::vector<comp> even_freqs, odd_freqs;
  
  //copy freqs in corresponding vectors
  for (unsigned int i = 0; i < freqs.size(); i++) {     
    //even or odd index
    if (i%2 == 0) {
      even_freqs.push_back(freqs[i]);
    } else {
      odd_freqs.push_back(freqs[i]);
    }
  }
  
  //run fft on new vecs
  std::vector<comp> even = fft(n/2, even_freqs);
  std::vector<comp> odd = fft(n/2, odd_freqs);
  
  //return array for values of fft
  std::vector<comp> c(n);

  for (unsigned int i = 0; i < n/2; i++) {
    //complex fft math. easy because complex cpp lib
    c[i] = even[i] + odd[i] * exp(comp(-2,0)*li*M_PI*comp(i,0)/comp(n,0));
    c[i+n/2] = even[i] - odd[i] * exp(comp(-2,0)*li*M_PI*comp(i,0)/comp(n,0));
  }

  return c;
}
