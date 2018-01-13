#include <sndfile.hh>
#include <cstdio>
#include <cstring>
#include <vector>

#define BUFFER_LEN 1024

std::vector<int> read_file(const char* fname);
