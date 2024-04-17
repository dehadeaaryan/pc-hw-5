#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAX_ARRAY_SIZE 100000

int cmpfunc(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

void PSRS(int *data, int n, int *pivots, int num_procs, int my_rank, MPI_Comm comm)
{
    // print the array
    printf("Process %d: Unsorted array:\n", my_rank);
    for (int i = 0; i < n; i++)
    {
        printf("%d ", data[i]);
    }
    printf("\n");
    printf("Process %d: Starting PSRS\n", my_rank);
    int local_size = n / num_procs;
    printf("Process %d: Local size: %d\n", my_rank, local_size);
    int *local_data = (int *)malloc(sizeof(int) * local_size);
    MPI_Scatter(data, local_size, MPI_INT, local_data, local_size, MPI_INT, 0, comm);

    // Step 1: Sort local data
    qsort(local_data, local_size, sizeof(int), (const void *)cmpfunc);

    // Step 2: Select p-1 equally spaced samples from all processes
    int *samples = (int *)malloc(sizeof(int) * (num_procs - 1));
    int *all_samples = (int *)malloc(sizeof(int) * num_procs * (num_procs - 1));
    for (int i = 0; i < num_procs - 1; i++)
    {
        samples[i] = local_data[(i + 1) * (local_size / (num_procs))];
    }
    MPI_Gather(samples, num_procs - 1, MPI_INT, all_samples, num_procs - 1, MPI_INT, 0, comm);

    // Step 3: Sort and select pivots
    if (my_rank == 0)
    {
        qsort(all_samples, num_procs * (num_procs - 1), sizeof(int), (const void *)cmpfunc);
        for (int i = 0; i < num_procs - 1; i++)
        {
            pivots[i] = all_samples[(i + 1) * (num_procs - 1)]; // Fix here
        }
    }
    MPI_Bcast(pivots, num_procs - 1, MPI_INT, 0, comm);

    // Step 4: Partition local data using pivots
    int *partition_sizes = (int *)malloc(sizeof(int) * num_procs);
    int *partition_disps = (int *)malloc(sizeof(int) * num_procs);
    int *send_counts = (int *)malloc(sizeof(int) * num_procs);
    int *recv_counts = (int *)malloc(sizeof(int) * num_procs);
    int *recv_disps = (int *)malloc(sizeof(int) * num_procs);
    int j = 0;
    for (int i = 0; i < num_procs - 1; i++)
    {
        int old_j = j;
        while (j < local_size && local_data[j] <= pivots[i])
            j++;
        send_counts[i] = j - old_j;
    }
    send_counts[num_procs - 1] = local_size - j;
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);
    partition_disps[0] = 0;
    for (int i = 1; i < num_procs; i++)
    {
        partition_disps[i] = partition_disps[i - 1] + recv_counts[i - 1];
    }
    int total_elements = partition_disps[num_procs - 1] + recv_counts[num_procs - 1];
    for (int i = 0; i < num_procs; i++)
    {
        partition_sizes[i] = total_elements / num_procs;
    }
    partition_sizes[num_procs - 1] = total_elements - partition_sizes[num_procs - 1] * (num_procs - 1);
    recv_disps[0] = 0;
    for (int i = 1; i < num_procs; i++)
    {
        recv_disps[i] = recv_disps[i - 1] + partition_sizes[i - 1];
    }
    int *new_local_data = (int *)malloc(sizeof(int) * total_elements);
    MPI_Alltoallv(local_data, send_counts, partition_disps, MPI_INT, new_local_data, recv_counts, recv_disps, MPI_INT, comm);

    // Step 5: Sort local data after partitioning
    qsort(new_local_data, total_elements, sizeof(int), (const void *)cmpfunc);

    // Step 6: Gather sorted data at root process
    int *recv_counts_gather = (int *)malloc(sizeof(int) * num_procs);
    int *recv_disps_gather = (int *)malloc(sizeof(int) * num_procs);
    MPI_Allgather(&partition_sizes[my_rank], 1, MPI_INT, recv_counts_gather, 1, MPI_INT, comm);
    recv_disps_gather[0] = 0;
    for (int i = 1; i < num_procs; i++)
    {
        recv_disps_gather[i] = recv_disps_gather[i - 1] + recv_counts_gather[i - 1];
    }
    if (my_rank == 0)
    {
        data = (int *)realloc(data, sizeof(int) * total_elements);
    }
    MPI_Gatherv(new_local_data, total_elements, MPI_INT, data, recv_counts_gather, recv_disps_gather, MPI_INT, 0, comm);

    free(local_data);
    free(samples);
    free(all_samples);
    free(partition_sizes);
    free(partition_disps);
    free(send_counts);
    free(recv_counts);
    free(recv_disps);
    free(new_local_data);
    free(recv_counts_gather);
    free(recv_disps_gather);
    printf("Process %d: PSRS finished\n", my_rank);
}

int main(int argc, char *argv[])
{
    int num_procs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc != 2)
    {
        if (my_rank == 0)
        {
            printf("Usage: %s <array_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int array_size;
    if (my_rank == 0)
    {
        array_size = atoi(argv[1]);
    }
    MPI_Bcast(&array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *data = (int *)malloc(sizeof(int) * array_size);
    int *pivots = (int *)malloc(sizeof(int) * (num_procs - 1));

    // Generate random data only in the root process
    if (my_rank == 0)
    {
        printf("Unsorted array:\n");
        for (int i = 0; i < array_size; i++)
        {
            data[i] = rand() % 1000; // Generate random data
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Execute PSRS algorithm
    PSRS(data, array_size, pivots, num_procs, my_rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (my_rank == 0)
    {
        printf("Sorted array:\n");
        for (int i = 0; i < array_size; i++)
        {
            printf("%d ", data[i]);
        }
        printf("\n");

        printf("Time taken: %f seconds\n", end_time - start_time);
    }

    free(data);
    free(pivots);
    MPI_Finalize();
    return 0;
}