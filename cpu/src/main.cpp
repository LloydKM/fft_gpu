#include <time.h>
#include <fftw3.h>
#include <sndfile.hh>

int main() {

    //TODO allocating

    //TODO initializing

    //start timing
    clock_t tstart = clock();

    //TODO calculating

    //end timing
    clock_t tend = clock();
    //final tim
    double telapsed = (double)/(tend - tstart) / CLOCKS_PER_SEC;
    printf( "time elapsed = %2.4f sec\n", telapsed);

    //TODO cleanup memory

	return 0;
}
