#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <vector>
#include <cstdlib>
#include <iostream>

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img, img_in.img, hist, result.w * result.h, 256);
    return result;
}

PPM_IMG contrast_enhancement_c_rgb(PPM_IMG img_in)
{
    PPM_IMG result;
    int hist[256];

    result.w = img_in.w;
    result.h = img_in.h;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    histogram(hist, img_in.img_r, img_in.h * img_in.w, 256);
    histogram_equalization(result.img_r, img_in.img_r, hist, result.w * result.h, 256);
    histogram(hist, img_in.img_g, img_in.h * img_in.w, 256);
    histogram_equalization(result.img_g, img_in.img_g, hist, result.w * result.h, 256);
    histogram(hist, img_in.img_b, img_in.h * img_in.w, 256);
    histogram_equalization(result.img_b, img_in.img_b, hist, result.w * result.h, 256);

    return result;
}

PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;

    unsigned char *y_equ;
    int hist[256];

    yuv_med = rgb2yuv(img_in);
    y_equ = (unsigned char *)malloc(yuv_med.h * yuv_med.w * sizeof(unsigned char));

    histogram(hist, yuv_med.img_y, yuv_med.h * yuv_med.w, 256);
    histogram_equalization(y_equ, yuv_med.img_y, hist, yuv_med.h * yuv_med.w, 256);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;

    result = yuv2rgb(yuv_med);
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);

    return result;
}

PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;

    unsigned char *l_equ;
    int hist[256];

    hsl_med = rgb2hsl(img_in);
    l_equ = (unsigned char *)malloc(hsl_med.height * hsl_med.width * sizeof(unsigned char));

    histogram(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);
    histogram_equalization(l_equ, hsl_med.l, hist, hsl_med.width * hsl_med.height, 256);

    free(hsl_med.l);
    hsl_med.l = l_equ;

    result = hsl2rgb(hsl_med);
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);
    return result;
}

// Convert RGB to HSL, assume R,G,B in [0, 255]
// Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG rgb2hsl(PPM_IMG img_in)
{
    HSL_IMG img_out;
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    img_out.width = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));
    int img_size = img_in.w * img_in.h;
    int local_size = img_size / world_size;
    unsigned char *local_img_r = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_g = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_b = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    float *local_h = (float *)malloc(local_size * sizeof(float));
    float *local_s = (float *)malloc(local_size * sizeof(float));
    unsigned char *local_l = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    MPI_Scatter(img_in.img_r, local_size, MPI_UNSIGNED_CHAR, local_img_r, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(img_in.img_g, local_size, MPI_UNSIGNED_CHAR, local_img_g, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(img_in.img_b, local_size, MPI_UNSIGNED_CHAR, local_img_b, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    for (int i = 0; i < local_size; ++i)
    {
        float var_r = (float)local_img_r[i] / 255;
        float var_g = (float)local_img_g[i] / 255;
        float var_b = (float)local_img_b[i] / 255;
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b; // min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b; // max. value of RGB
        float del_max = var_max - var_min;             // Delta RGB value
        float L = (var_max + var_min) / 2;
        float H, S;
        if (del_max == 0)
        {
            H = 0;
            S = 0;
        }
        else
        {
            if (L < 0.5)
                S = del_max / (var_max + var_min);
            else
                S = del_max / (2 - var_max - var_min);
            float del_r = (((var_max - var_r) / 6) + (del_max / 2)) / del_max;
            float del_g = (((var_max - var_g) / 6) + (del_max / 2)) / del_max;
            float del_b = (((var_max - var_b) / 6) + (del_max / 2)) / del_max;
            if (var_r == var_max)
                H = del_b - del_g;
            else if (var_g == var_max)
                H = (1.0 / 3.0) + del_r - del_b;
            else
                H = (2.0 / 3.0) + del_g - del_r;
            if (H < 0)
                H += 1;
            if (H > 1)
                H -= 1;
        }
        local_h[i] = H;
        local_s[i] = S;
        local_l[i] = (unsigned char)(L * 255);
    }
    MPI_Gather(local_h, local_size, MPI_FLOAT, img_out.h, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_s, local_size, MPI_FLOAT, img_out.s, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_l, local_size, MPI_UNSIGNED_CHAR, img_out.l, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    free(local_img_r);
    free(local_img_g);
    free(local_img_b);
    free(local_h);
    free(local_s);
    free(local_l);

    return img_out;
}
float Hue_2_RGB(float v1, float v2, float vH) // Function Hue_2_RGB
{
    if (vH < 0)
        vH += 1;
    if (vH > 1)
        vH -= 1;
    if ((6 * vH) < 1)
        return (v1 + (v2 - v1) * 6 * vH);
    if ((2 * vH) < 1)
        return (v2);
    if ((3 * vH) < 2)
        return (v1 + (v2 - v1) * ((2.0f / 3.0f) - vH) * 6);
    return (v1);
}

// Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
// Output R,G,B in [0, 255]
PPM_IMG hsl2rgb(HSL_IMG img_in)
{
    PPM_IMG result;
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    int img_size = img_in.width * img_in.height;
    int local_size = img_size / world_size;
    unsigned char *local_img_r = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_g = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_b = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    float *local_h = (float *)malloc(local_size * sizeof(float));
    float *local_s = (float *)malloc(local_size * sizeof(float));
    unsigned char *local_l = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    MPI_Scatter(img_in.h, local_size, MPI_FLOAT, local_h, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(img_in.s, local_size, MPI_FLOAT, local_s, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(img_in.l, local_size, MPI_UNSIGNED_CHAR, local_l, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    for (int i = 0; i < local_size; ++i)
    {
        float H = local_h[i];
        float S = local_s[i];
        float L = local_l[i] / 255.0f;
        float var_1, var_2;
        unsigned char r, g, b;
        if (S == 0)
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {
            if (L < 0.5)
                var_2 = L * (1 + S);
            else
                var_2 = (L + S) - (S * L);
            var_1 = 2 * L - var_2;
            r = 255 * Hue_2_RGB(var_1, var_2, H + (1.0f / 3.0f));
            g = 255 * Hue_2_RGB(var_1, var_2, H);
            b = 255 * Hue_2_RGB(var_1, var_2, H - (1.0f / 3.0f));
        }
        local_img_r[i] = r;
        local_img_g[i] = g;
        local_img_b[i] = b;
    }
    MPI_Gather(local_img_r, local_size, MPI_UNSIGNED_CHAR, result.img_r, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(local_img_g, local_size, MPI_UNSIGNED_CHAR, result.img_g, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(local_img_b, local_size, MPI_UNSIGNED_CHAR, result.img_b, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    free(local_img_r);
    free(local_img_g);
    free(local_img_b);
    free(local_h);
    free(local_s);
    free(local_l);

    return result;
}
// Convert RGB to YUV, all components in [0, 255]
YUV_IMG rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char) * img_out.w * img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char) * img_out.w * img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char) * img_out.w * img_out.h);

    int img_size = img_out.w * img_out.h;
    int local_size = img_size / world_size;

    unsigned char *local_img_r = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_g = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_b = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_y = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_u = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_v = (unsigned char *)malloc(local_size * sizeof(unsigned char));

    MPI_Scatter(img_in.img_r, local_size, MPI_UNSIGNED_CHAR, local_img_r, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(img_in.img_g, local_size, MPI_UNSIGNED_CHAR, local_img_g, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(img_in.img_b, local_size, MPI_UNSIGNED_CHAR, local_img_b, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_size; ++i)
    {
        unsigned char r = local_img_r[i];
        unsigned char g = local_img_g[i];
        unsigned char b = local_img_b[i];

        unsigned char y = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        unsigned char cb = (unsigned char)(-0.169 * r - 0.331 * g + 0.499 * b + 128);
        unsigned char cr = (unsigned char)(0.499 * r - 0.418 * g - 0.0813 * b + 128);

        local_img_y[i] = y;
        local_img_u[i] = cb;
        local_img_v[i] = cr;
    }

    MPI_Gather(local_img_y, local_size, MPI_UNSIGNED_CHAR, img_out.img_y, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(local_img_u, local_size, MPI_UNSIGNED_CHAR, img_out.img_u, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(local_img_v, local_size, MPI_UNSIGNED_CHAR, img_out.img_v, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    free(local_img_r);
    free(local_img_g);
    free(local_img_b);
    free(local_img_y);
    free(local_img_u);
    free(local_img_v);

    return img_out;
}

unsigned char clip_rgb(int x)
{
    if (x > 255)
        return 255;
    if (x < 0)
        return 0;

    return (unsigned char)x;
}

// Convert YUV to RGB, all components in [0, 255]
PPM_IMG yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char) * img_out.w * img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char) * img_out.w * img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char) * img_out.w * img_out.h);

    int img_size = img_out.w * img_out.h;
    int local_size = img_size / world_size;

    unsigned char *local_img_y = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_u = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_v = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_r = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_g = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    unsigned char *local_img_b = (unsigned char *)malloc(local_size * sizeof(unsigned char));

    MPI_Scatter(img_in.img_y, local_size, MPI_UNSIGNED_CHAR, local_img_y, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(img_in.img_u, local_size, MPI_UNSIGNED_CHAR, local_img_u, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(img_in.img_v, local_size, MPI_UNSIGNED_CHAR, local_img_v, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_size; ++i)
    {
        int y = (int)local_img_y[i];
        int cb = (int)local_img_u[i] - 128;
        int cr = (int)local_img_v[i] - 128;

        int rt = (int)(y + 1.402 * cr);
        int gt = (int)(y - 0.344 * cb - 0.714 * cr);
        int bt = (int)(y + 1.772 * cb);

        local_img_r[i] = clip_rgb(rt);
        local_img_g[i] = clip_rgb(gt);
        local_img_b[i] = clip_rgb(bt);
    }

    MPI_Gather(local_img_r, local_size, MPI_UNSIGNED_CHAR, img_out.img_r, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(local_img_g, local_size, MPI_UNSIGNED_CHAR, img_out.img_g, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(local_img_b, local_size, MPI_UNSIGNED_CHAR, img_out.img_b, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    free(local_img_y);
    free(local_img_u);
    free(local_img_v);
    free(local_img_r);
    free(local_img_g);
    free(local_img_b);

    return img_out;
}
