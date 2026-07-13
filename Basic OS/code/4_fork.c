#include <stdio.h>
#include <time.h>
#include <unistd.h>

int main() {
	// Create timespec structs to hold start & end times of computation
	struct timespec tstart = {0, 0}, tend = {0, 0};

	// Get time from OS
	clock_gettime(CLOCK_MONOTONIC, &tstart);

	fork();
	fork();

	for (int i = 0; i < 2147483647; i++) asm("nop");

	// Get time again
	clock_gettime(CLOCK_MONOTONIC, &tend);

	// Print time difference between tstart and tend
	printf("loop took about %.5f seconds\n",
			((double)tend.tv_sec + 1.0e-9 * tend.tv_nsec) - 
			((double)tstart.tv_sec + 1.0e-9 * tstart.tv_nsec));

	return 0;
}

