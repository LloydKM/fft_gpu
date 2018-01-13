#include <sndfile.hh>
#include <iostream>
#include <cstring>
#include "sndread.h"

std::vector<int> read_file(const char* fname) {

  SndfileHandle haendel = SndfileHandle(fname);

  printf ("Opened file '%s'\n", fname) ;

  std::cout << "Frames: " << haendel.frames() << std::endl;
  std::cout << "Channels: " << haendel.channels() << std::endl;

  int buffer_length = haendel.channels() * haendel.frames();
  int buffer[BUFFER_LEN];

  std::vector<int> ret;
  while(haendel.read(buffer, BUFFER_LEN) != 0){

    for (auto const& c : buffer ) {
      ret.push_back(c);
    }
  }
  puts ("") ;

  return ret;
}
