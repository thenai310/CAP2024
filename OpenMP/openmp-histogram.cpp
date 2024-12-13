#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>

void histogram(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin)
{
// Initialize histogram
#pragma omp parallel sections
    {
#pragma omp section
        {
#pragma omp parallel for
            for (int i = 0; i < nbr_bin / 2; ++i)
            {
                hist_out[i] = 0;
            }
        }

#pragma omp section
        {
#pragma omp parallel for
            for (int i = nbr_bin / 2; i < nbr_bin; ++i)
            {
                hist_out[i] = 0;
            }
        }
    }

// Compute histogram
#pragma omp parallel for
    for (int i = 0; i < img_size; ++i)
    {
#pragma omp atomic
        hist_out[img_in[i]]++;
    }
}

void histogram_equalization(unsigned char *img_out, unsigned char *img_in, int *hist_in, int img_size, int nbr_bin)
{
    std::vector<int> lut(nbr_bin, 0);
    int cdf = 0, min = 0, d;

// Find the minimum non-zero value in the histogram
#pragma omp parallel sections
    {
#pragma omp section
        {
#pragma omp parallel for reduction(min : min)
            for (int i = 0; i < nbr_bin / 2; ++i)
            {
                if (hist_in[i] != 0)
                {
                    min = hist_in[i];
                }
            }
        }

#pragma omp section
        {
#pragma omp parallel for reduction(min : min)
            for (int i = nbr_bin / 2; i < nbr_bin; ++i)
            {
                if (hist_in[i] != 0)
                {
                    min = hist_in[i];
                }
            }
        }
    }

    d = img_size - min;

// Compute the LUT (Look-Up Table)
#pragma omp parallel sections
    {
#pragma omp section
        {
#pragma omp parallel for reduction(+ : cdf)
            for (int i = 0; i < nbr_bin / 2; ++i)
            {
                cdf += hist_in[i];
                lut[i] = (int)(((float)cdf - min) * 255 / d + 0.5);
                if (lut[i] < 0)
                {
                    lut[i] = 0;
                }
            }
        }

#pragma omp section
        {
#pragma omp parallel for reduction(+ : cdf)
            for (int i = nbr_bin / 2; i < nbr_bin; ++i)
            {
                cdf += hist_in[i];
                lut[i] = (int)(((float)cdf - min) * 255 / d + 0.5);
                if (lut[i] < 0)
                {
                    lut[i] = 0;
                }
            }
        }
    }

// Apply the LUT to the image
#pragma omp parallel for
    for (int i = 0; i < img_size; ++i)
    {
        img_out[i] = (lut[img_in[i]] > 255) ? 255 : (unsigned char)lut[img_in[i]];
    }
}