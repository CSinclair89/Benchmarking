#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>

#define MSG_SIZE 64000 // msg size for latency test
#define ITERATIONS 100 // No. iterations for latency test
#define DATA_SIZE 64000 // byte size for thoughput test

// Measure latency
void measureLatency(int sockfd) {
	char msg[MSG_SIZE]; // declare char array of size MSG_SIZE
	struct timeval tic, toc; // declare timeval variables
	long totalTime = 0; // initialize total at 0

	for (int i = 0; i < ITERATIONS; i++) { // loop through up to ITERATIONS
		gettimeofday(&tic, NULL); // parameter one holds returned time, paramater two holds timezone (usually just stays NULL)

		// send msg from parent to child
		write(sockfd, msg, MSG_SIZE); // write the message into the socket, character array of 'msg', and size of msg

		// read response from child
		read(sockfd, msg, MSG_SIZE); // reads the msg from child through socket, char array of 'msg', and size of msg

		gettimeofday(&toc, NULL); // parameter one holds returned time, param two holds timezone (usually stays NULL)

		long timeElapsed = (toc.tv_sec - tic.tv_sec) * 1000000 + (toc.tv_usec - tic.tv_usec); // run calculation by subtracting start time from end time, multiply by 1000000, and then add the same calculation. first calculation is for seconds (tv_sec), second calculation is for microseconds (tv_usec)

		totalTime += timeElapsed; // add timeElapsed to totalTime
	}

	double avgLatency = (double)totalTime / ITERATIONS; // take mean of total time with iteration count
	printf("Avg latency: %.3f microseconds\n", avgLatency); // print the output statement
}


// Measure throughput
void measureThroughput (int sockfd) {
	char *data = malloc(DATA_SIZE);
	memset(data, 'A', DATA_SIZE);

	struct timeval tic, toc;
	gettimeofday(&tic, NULL);

	write(sockfd, data, DATA_SIZE);

	read(sockfd, data, DATA_SIZE);

	gettimeofday(&toc, NULL);

	long elapsedTime = (toc.tv_sec - tic.tv_sec) * 1000000 + (toc.tv_usec - tic.tv_usec);

	double throughput = (double)DATA_SIZE / (elapsedTime / 1000000.0) / (1024 * 1024);

	printf("Throughput: %.3f MB/s\n", throughput);

	free(data);
}


// Main method
int main() {

	int sv[2]; // socket pair
	
	// Create socketpair for communication
	
	if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == -1) {
		perror("socketpair");
		exit(1);
	}

	pid_t pid = fork();
	if (pid == -1) {
		perror("fork");
		exit(1);
	}

	if (pid == 0) { // Child process
		close(sv[0]); // Close parent socket
		char msg[MSG_SIZE];

		// For latency test, just echo back what we receive
		for (int i = 0; i < ITERATIONS; i++) {
			read(sv[1], msg, MSG_SIZE);
			write(sv[1], msg, MSG_SIZE);
		}

		// For throughput test, receive data and send a 1-byte ack
		char *data = malloc(DATA_SIZE);
		read(sv[1], data, DATA_SIZE);
		write(sv[1], "1", 1);

		free(data);
		close(sv[1]);
	} else { // parent process
		close(sv[1]); // close child socket
		
		printf("Measuring latency...\n");
		measureLatency(sv[0]);

		printf("Measuring throughput...\n");
		measureThroughput(sv[0]);

		close(sv[0]);
	}

	return 0;
}

























