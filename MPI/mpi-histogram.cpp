#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

void histogram(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Initialize local histogram
    std::vector<int> local_hist(nbr_bin, 0);

    // Determine the portion of the image each process will handle
    int local_size = img_size / world_size;
    int start = world_rank * local_size;
    int end = (world_rank == world_size - 1) ? img_size : start + local_size;

    // Compute local histogram
    for (int i = start; i < end; ++i) {
        local_hist[img_in[i]]++;
    }

    // Gather all local histograms to the root process
    MPI_Reduce(local_hist.data(), hist_out, nbr_bin, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Broadcast the final histogram to all processes
    MPI_Bcast(hist_out, nbr_bin, MPI_INT, 0, MPI_COMM_WORLD);
}

void histogram_equalization(unsigned char *img_out, unsigned char *img_in, int *hist_in, int img_size, int nbr_bin) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Initialize local histogram
    std::vector<int> local_hist(nbr_bin, 0);

    // Determine the portion of the image each process will handle
    int local_size = img_size / world_size;
    int start = world_rank * local_size;
    int end = (world_rank == world_size - 1) ? img_size : start + local_size;

    // Compute local histogram
    for (int i = start; i < end; ++i) {
        local_hist[img_in[i]]++;
    }

    // Gather all local histograms to the root process
    std::vector<int> global_hist(nbr_bin, 0);
    MPI_Reduce(local_hist.data(), global_hist.data(), nbr_bin, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Broadcast the global histogram to all processes
    MPI_Bcast(global_hist.data(), nbr_bin, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the LUT (Look-Up Table) on the root process
    std::vector<int> lut(nbr_bin, 0);
    if (world_rank == 0) {
        int cdf = 0, min = 0, d;
        for (int i = 0; i < nbr_bin; ++i) {
            if (global_hist[i] != 0) {
                min = global_hist[i];
                break;
            }
        }
        d = img_size - min;
        for (int i = 0; i < nbr_bin; ++i) {
            cdf += global_hist[i];
            lut[i] = (int)(((float)cdf - min) * 255 / d + 0.5);
            if (lut[i] < 0) {
                lut[i] = 0;
            }
        }
    }

    // Broadcast the LUT to all processes
    MPI_Bcast(lut.data(), nbr_bin, MPI_INT, 0, MPI_COMM_WORLD);

    // Apply the LUT to the image
    for (int i = start; i < end; ++i) {
        img_out[i] = (lut[img_in[i]] > 255) ? 255 : (unsigned char)lut[img_in[i]];
    }

    // Gather the processed image parts to the root process
    MPI_Gather(img_out + start, local_size, MPI_UNSIGNED_CHAR, img_out, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

