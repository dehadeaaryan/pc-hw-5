#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void print_array(int *array, int size, int rank) {
    printf("Process %d: ", rank);
    for (int i = 0; i < size; i++)
        printf("%d ", array[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_size = 10; 
    int *data = (int *)malloc(local_size * sizeof(int));
    for (int i = 0; i < local_size; i++) {
        data[i] = rand() % 100;
    }

    qsort(data, local_size, sizeof(int), compare);

    int *samples = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        samples[i] = data[local_size / size * (i + 1) - 1];
    }

    int *gathered_samples;
    if (rank == 0) {
        gathered_samples = (int *)malloc(size * size * sizeof(int));
    }
    MPI_Gather(samples, size, MPI_INT, gathered_samples, size, MPI_INT, 0, MPI_COMM_WORLD);

    int *pivots;
    if (rank == 0) {
        qsort(gathered_samples, size * size, sizeof(int), compare);
        pivots = (int *)malloc((size - 1) * sizeof(int));
        for (int i = 0; i < size - 1; i++) {
            pivots[i] = gathered_samples[(i + 1) * size + size / 2];
        }
    }

    if (rank != 0) {
        pivots = (int *)malloc((size - 1) * sizeof(int));
    }
    MPI_Bcast(pivots, size - 1, MPI_INT, 0, MPI_COMM_WORLD);
    print_array(data, local_size, rank);
    if (rank == 0) {
        printf("Pivots: ");
        for (int i = 0; i < size - 1; i++) {
            printf("%d ", pivots[i]);
        }
        printf("\n");
    }

    free(data);
    free(samples);
    if (rank == 0) {
        free(gathered_samples);
        free(pivots);
    } else {
        free(pivots);
    }

    MPI_Finalize();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

/*
PSRS STEPS
Step 1: Input data to be sorted of size N is initially portioned among the P processes so that each process gets N/P items to sort. Each process initially sorts its own sub-list of N/P items using sequential quick sort.
• Step 2: Each process Pi selects P items (we call them regular samples) from its local sorted sub-list at the following local indices: [0, N/P², 2N/P²,…, (P-1)N/P² ] and sends them to process P₀.
• Step 3: Process P₀ collects the regular samples from the P processes (which includes itself). So it has P² total regular samples. It sorts the regular samples using quick sort, and then chooses (P-1) pivot values at the following indices: [P + P/2-1, 2P + P/2 -1,…, (P-1)P + P/2 -1] and broadcasts the pivots to the P processes.
• Step 4: Each process Pᵢ, upon receiving the P-1 pivots from process P₀, partitions its local sorted sub-list into P partitions, with the P-1 pivots as separators. Then it keeps ith partition for itself and sends jth partition to process Pⱼ.
• Step 5: At the end of step 4, each process Pᵢ has (P-1) partitions from other processes together with its own ith partition. It locally merges all P (sorted) partitions to create its final sorted sub-list.
*/

// Function to compare integers (for qsort)
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

// Function to perform PSRS algorithm
void psrs(int *data, int n, int *sorted_data) {
    // Step 1: Quicksort local data
    qsort(data, n, sizeof(int), compare);

    // Step 2: Select local regular samples
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Size: %d, Rank: %d\n", size, rank);
    int *samples = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        samples[i] = data[i * n / (size * size)];
    }
    
    // Step 3: Gather all samples
    int *all_samples = NULL;
    if (rank == 0) {
        all_samples = (int *)malloc(size * size * sizeof(int));
    }
    MPI_Gather(samples, size, MPI_INT, all_samples, size, MPI_INT, 0, MPI_COMM_WORLD);
    // sort all_samples using quicksort, choose pivots at [P + P/2-1, 2P + P/2 -1,…, (P-1)P + P/2 -1] and broadcast to all processes
    if (rank == 0) {
        qsort(all_samples, size*size, sizeof(int), compare);
        for (int i = 0; i < size - 1; i++) {
            sorted_data[i] = all_samples[(i + 1) * size + size / 2 - 1];
        }
    }
    MPI_Bcast(sorted_data, size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 4: Select global regular samples
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int i = 0, j = 0; i < n; i++) {
        if (j < size - 1 && data[i] > sorted_data[j]) {
            sendcounts[j] = i - displs[j];
            displs[++j] = i;
        }
    }
    sendcounts[size - 1] = n - displs[size - 1];

    int *recvcounts = (int *)malloc(size * sizeof(int));
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    int total = 0;
    for (int i = 0; i < size; i++) {
        total += recvcounts[i];
    }
    int *sorted_sublist = (int *)malloc(total * sizeof(int));

    MPI_Alltoallv(data, sendcounts, displs, MPI_INT, sorted_sublist, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);

    // Step 5: Merge all received partitions to create the final sorted sublist
    qsort(sorted_sublist, total, sizeof(int), compare);
    memcpy(data, sorted_sublist, total * sizeof(int));

    free(samples);
    free(all_samples);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(sorted_sublist);
}

int main(int argc, char *argv[]) {

    int rank, size;
    int *data = NULL;
    int *sorted_data = NULL;
   
   if (argc != 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {

        data = (int *)malloc(n * sizeof(int));
        // Initialize or read data
        printf("Unsorted Data: ");
        for (int i = 0; i < n; i++) {
            data[i] = rand() % 1000;
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    // Broadcast the size of the problem to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for scattered data
    int *scattered_data = (int *)malloc(n / size * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);

    // Scatter data to each processor
    MPI_Scatter(data, n / size, MPI_INT, scattered_data, n / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort scattered data individually on each processor
    qsort(scattered_data, n / size, sizeof(int), compare);

    // Allocate memory for sorted data on master process
    if (rank == 0) {
        sorted_data = (int *)malloc(n * sizeof(int));
    }

    // Gather sorted data to master process
    MPI_Gather(scattered_data, n / size, MPI_INT, sorted_data, n / size, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d: ", rank);
    for (int i = 0; i < n / size; i++) {
        printf("%d ", scattered_data[i]);
    }
    printf("\n");

    printf("Process %d: PSRS finished\n", rank);
    printf("Scattered Data: ");
    for (int i = 0; i < n / size; i++) {
        printf("%d ", scattered_data[i]);
    }
    printf("\n");

    // Merge sorted data on master process
    if (rank == 0) {
        // Perform PSRS algorithm on merged sorted data
        psrs(sorted_data, n, sorted_data);

        // Output sorted data
        printf("Sorted Data: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", sorted_data[i]);
        }
        printf("\n");

        free(data);
        free(sorted_data);
    }

    free(scattered_data);
    MPI_Finalize();
    return 0;
}