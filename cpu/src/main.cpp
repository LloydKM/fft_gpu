#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <complex>
#include <iomanip>

#include "sndread.h"

typedef std::complex<double> comp;
#define li comp(0,1)
#define F_PATH "../src/briOPETH.wav"

std::vector<comp> pad_with_zero(std::vector<comp> v);
std::vector<comp> fft(std::vector<comp> freqs);
std::vector<comp> _fft(int n, std::vector<comp> freqs);

int main(int argc, char** argv) {
  std::vector<int> test = read_file(F_PATH);
  std::vector<comp> v;
  for (auto const& val: test) {
    v.push_back(comp(val,0));
  }
  std::vector<comp> fftv = fft(v);
  for (auto const& value: fftv) {
    std::cout << value << std::endl;
  }
}

std::vector<comp> pad_with_zero(std::vector<comp> v) {
  int new_size = 0;
  int exponent = 0;
  //get next greater power of two
  while (new_size < v.size()) {
    new_size = pow(2, exponent++);
  }

  //pad vector with zeroes
  int v_size = v.size();
  for (int i = 0; i < (new_size - v_size); i++) {
    v.push_back(comp(0,0));
  }
  return v;
}

std::vector<comp> fft(std::vector<comp> freqs) {
  int n = freqs.size();
  //check if length is power of two
  if (n & (n-1) == 0) {
    return _fft(n, freqs);
  } else {
    //padding with zeroes
    std::cout << "padding" << std::endl;
    std::vector<comp> padded = pad_with_zero(freqs);
    std::cout << std::endl;
    return _fft(padded.size(), padded);
  }

}

std::vector<comp> _fft(int n, std::vector<comp> freqs) {
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
  std::vector<comp> even = _fft(n/2, even_freqs);
  std::vector<comp> odd = _fft(n/2, odd_freqs);

  //return array for values of fft
  std::vector<comp> c(n);

  for (unsigned int i = 0; i < n/2; i++) {
    //complex fft math. easy because complex cpp lib
    c[i] = even[i] + odd[i] * exp(comp(-2,0)*li*M_PI*comp(i,0)/comp(n,0));
    c[i+n/2] = even[i] - odd[i] * exp(comp(-2,0)*li*M_PI*comp(i,0)/comp(n,0));
  }

  return c;
}
