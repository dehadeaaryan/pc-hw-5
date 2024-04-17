#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

// Function to compare integers (for qsort)
int compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

void printArray(int arr[], int size)
{
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
}

int main(int argc, char *argv[])
{

    int rank, size;
    int *data = NULL;
    int *sorted_data = NULL;

    if (argc != 2)
    {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {

        data = (int *)malloc(n * sizeof(int));
        // Initialize or read data
        printf("Unsorted Data: ");
        for (int i = 0; i < n; i++)
        {
            data[i] = rand() % 1000;
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    // Broadcast the size of the problem to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for scattered data in all processes
    int *scattered_data = (int *)malloc(n / size * sizeof(int));

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Scatter data to each processor
    MPI_Scatter(data, n / size, MPI_INT, scattered_data, n / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort scattered data individually on each processor
    qsort(scattered_data, n / size, sizeof(int), compare);

    // get size^2 samples; size samples from each process
    int *samples = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        samples[i] = scattered_data[i * n / (size * size)];
    }

    // Gather all samples to process 0
    int *all_samples = NULL;
    if (rank == 0)
    {
        all_samples = (int *)malloc(size * size * sizeof(int));
    }
    MPI_Gather(samples, size, MPI_INT, all_samples, size, MPI_INT, 0, MPI_COMM_WORLD);

    // merge sort all samples
    if (rank == 0)
    {
        qsort(all_samples, 0, size * size - 1, compare);
    }

    // broadcast pivots to all processes
    int *pivots = (int *)malloc(size - 1);
    if (rank == 0)
    {
        for (int i = 0; i < size - 1; i++)
        {
            pivots[i] = all_samples[(i + 1) * size + size / 2 - 1];
        }
    }
    MPI_Bcast(pivots, size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // send data to processes based on pivots
    // Initialize sendcounts, displs, and recvcounts to 0
    int *sendcounts = (int *)calloc(size, sizeof(int));
    int *displs = (int *)calloc(size, sizeof(int));
    int *recvcounts = (int *)calloc(size, sizeof(int));

    // Calculate sendcounts and displs
    for (int i = 0, j = 0; i < n / size; i++)
    {
        if (j < size - 1 && scattered_data[i] > pivots[j])
        {
            sendcounts[j] = i - displs[j];
            displs[++j] = i;
        }
    }
    sendcounts[size - 1] = n / size - displs[size - 1];

    // Use MPI_Alltoall to get recvcounts
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate total and displs for MPI_Alltoallv
    int total = 0;
    for (int i = 0; i < size; i++)
    {
        total += recvcounts[i];
        if (i > 0)
        {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }
    }

    int *sorted_sublist = (int *)malloc(total * sizeof(int));

    // Use MPI_Alltoallv to exchange data
    MPI_Alltoallv(scattered_data, sendcounts, displs, MPI_INT, sorted_sublist, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);

    // merge sort all received partitions
    qsort(sorted_sublist, 0, total - 1, compare);

    // Print sorted data
    printf("\n");
    printArray(sorted_sublist, total);

    // Send sorted data back to process 0
    // MPI_Gather(sorted_sublist, total, MPI_INT, data, total, MPI_INT, 0, MPI_COMM_WORLD);

    // // Print sorted data
    // if (rank == 0)
    // {
    //     printf("Sorted Data: ");
    //     for (int i = 0; i < n; i++)
    //     {
    //         printf("%d ", data[i]);
    //     }
    //     printf("\n");
    // }

    // free memory
    free(samples);
    free(all_samples);
    free(pivots);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(sorted_sublist);
    free(scattered_data);
    MPI_Finalize();
    printf("\n");
    return 0;
}
