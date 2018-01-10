#include <sndfile.hh>
#include <iostream>
#include <cstring>
#include "sndread.h"

#define BUFFER_LEN 8

void read_file(const char* fname);

void read_file(const char* fname) {

  static int buffer[BUFFER_LEN];

  SndfileHandle haendel = SndfileHandle(fname);

  printf ("Opened file '%s'\n", fname) ;
  printf ("    Sample rate : %d\n", haendel.samplerate ()) ;
  printf ("    Channels    : %d\n", haendel.channels ()) ;
  
  for (int i = 0; i < 2; i++) {
    haendel.read(buffer, BUFFER_LEN);
    for (auto const& val: buffer) {
      std::cout << val << std::endl;
    }
  }

  puts ("") ;

}
