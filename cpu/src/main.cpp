#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>
#include <sndfile.h>

#define BUFFER_SIZE 1024

//Linux implementation

int main(int argc, char** argv) {

    //set SF_INFO to properly open file to read
    SF_INFO read_info;
    read_info.format = 0;

    char* fname;
    int* buffer = (int*) malloc(BUFFER_SIZE * sizeof(int));

    //use test file if none is passed
    if( argc < 2) {
        fname = (char*) malloc(16 * sizeof(char));
        strcpy(fname, "../src/test.wav");
    } else {
        fname = (char*) malloc((strlen(argv[1])+1) * sizeof(char));
        strcpy(fname, argv[1]);
    }

    SNDFILE* input = sf_open(fname, SFM_READ, &read_info);
    if (input == NULL){
        printf("YOU FAIL!\n");
        return -1;
    }

    //start timing
    clock_t tstart = clock();

    //TODO calculating
    //fftw
    #if 1
    while(sf_readf_int(input, buffer, BUFFER_SIZE) != 0){
        //printf("%d\n", buffer[0] );
    }

    //own fft
    #else

    #endif

    //end timing
    clock_t tend = clock();
    //final tim
    double telapsed = (double)(tend - tstart) / CLOCKS_PER_SEC;
    printf( "time elapsed = %2.4f sec\n", telapsed);

    //TODO cleanup memory
    free(buffer);
    free(fname);
    if (sf_close(input) != 0) printf("cleanup failed!\n");

	return 0;
}
