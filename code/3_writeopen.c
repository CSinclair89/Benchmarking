#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

int main() {

	int fd;
	char *msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas varius justo vitae est eleifend interdum. Morbi sit amet faucibus augue, nec laoreet mi. Nullam eu accumsan eros. Vestibulum mattis ligula nunc, a molestie risus faucibus et. Praesent eu bibendum lacus, quis egestas risus. Aenean facilisis enim eu ipsum finibus pharetra. Vivamus in tristique ipsum. Proin finibus nisi eu ex iaculis consectetur.";

	// Create timespec structs to hold start & end times of computation
	struct timespec tstart = {0, 0}, tend = {0, 0};

	// Get time from OS
	clock_gettime(CLOCK_MONOTONIC, &tstart);

	fd = open("LoremIpsum1.txt", O_WRONLY | O_CREAT);
	ssize_t bytes_written = write(fd, msg, strlen(msg));

	// Get time again
	clock_gettime(CLOCK_MONOTONIC, &tend);

	// Print time difference between tstart and tend
	printf("loop took about %.5f seconds\n",
			((double)tend.tv_sec + 1.0e-9 * tend.tv_nsec) - 
			((double)tstart.tv_sec + 1.0e-9 * tstart.tv_nsec));

	return 0;
}

