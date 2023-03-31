#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8




#define CC_GRID_MAX_X_DIM (1 << 31) - 1
#define CC_GRID_MAX_Y_DIM 65535
#define CC_GRID_MAX_Z_DIM 65535

#define CC_BLOCK_MAX_X_DIM 1024
#define CC_BLOCK_MAX_Y_DIM 1024
#define CC_BLOCK_MAX_Z_DIM 64

#define CC_BLOCK_MAX_N_THREADS 1024

#define CC_MAX_N_RSD_BLOCKS 32
#define CC_MAX_N_RSD_WARPS 64
#define CC_MAX_N_RSD_THREADS 2048


// 1.png: 4928 x 3264 x 3
// 2.png: 16320 x 10809 x 3
// 3.png: 634 x 634 x 3

// 4.png: 900 x 622 x 3

// 5.png: 1800 x 1244 x 3
// 6.png: 3600 x 2488 x 3
// 7.png: 7200 x 4976 x 3
// 8.png: 14400 x 9952 x 3


// #define GRID_N_X
// #define GRID_N_Y
#define GRID_N_Z 3

#define BLOCK_N_X 8
#define BLOCK_N_Y 8
#define BLOCK_N_Z 1

// #define BLOCK_N_THREADS




int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width,
    unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);


    // printf("rowbytes: %d\n", rowbytes);
    // printf("(rowbytes + 4*3) * (*height + 4): %d\n", (rowbytes + 4*3) * (*height + 4));
    
    if ((*image = (unsigned char*)calloc((rowbytes + 4*3) * (*height + 4), 1)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }
 
    for (i = 0; i < *height; ++i) {
        row_pointers[i] = *image + (i + 2) * (rowbytes + 4*3) + 2*3;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}


void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
    const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}



// __global__ void sobel(unsigned char *s, unsigned char *t, 
//                                 unsigned height, unsigned width, unsigned channels)
// {

//     int mask[MASK_N][MASK_X][MASK_Y] = {
    
//         {{ -1, -4, -6, -4, -1},
//         { -2, -8,-12, -8, -2},
//         {  0,  0,  0,  0,  0},
//         {  2,  8, 12,  8,  2},
//         {  1,  4,  6,  4,  1}},

//         {{ -1, -2,  0,  2,  1},
//         { -4, -8,  0,  8,  4},
//         { -6,-12,  0, 12,  6},
//         { -4, -8,  0,  8,  4},
//         { -1, -2,  0,  2,  1}}

//     };

//     int basex = blockIdx.x * blockDim.x;
//     int basey = blockIdx.y * blockDim.y;
    
//     int nextBasex = (blockIdx.x + 1) * blockDim.x;
//     int nextBasey = (blockIdx.y + 1) * blockDim.y;

//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int z = blockIdx.z * blockDim.z + threadIdx.z;


//     // printf("threadIdx.x: %d, blockIdx.x: %d, blockDim.x: %d\n", threadIdx.x, blockIdx.x, blockDim.x);

//     if(x >= width || y >= height) return;



//     __shared__ unsigned char smSrc[(BLOCK_N_X + 4) * (BLOCK_N_Y + 4)];

//     smSrc[(BLOCK_N_X + 4) * threadIdx.y + threadIdx.x] = s[3 * ((width + 4) * y + x) + z];
    
//     if(threadIdx.x < 4){
//         smSrc[(BLOCK_N_X + 4) * threadIdx.y + threadIdx.x + BLOCK_N_X] =\
//                     s[3 * ((width + 4) * (basey + threadIdx.y) + nextBasex + threadIdx.x) + z];
//     }
    
//     if(threadIdx.y < 4){
//         smSrc[(BLOCK_N_X + 4) * (BLOCK_N_Y + threadIdx.y) + threadIdx.x] =\
//                     s[3 * ((width + 4) * (nextBasey + threadIdx.y) + basex + threadIdx.x) + z];
//     }

//     if(threadIdx.x < 4 && threadIdx.y < 4){
//         smSrc[(BLOCK_N_X + 4) * (BLOCK_N_Y + threadIdx.y) + BLOCK_N_X + threadIdx.x] = \
//                     s[3 * ((width + 4) * (nextBasey + threadIdx.y) + nextBasex + threadIdx.x) + z];
//     }

    

//     __syncthreads();

    

//     double val[MASK_N] = {0.0};
    
//     for (int i = 0; i < MASK_N; ++i){
//         for (int v = 0; v < 5; ++v){
//             for (int u = 0; u < 5; ++u){
//                 val[i] += ((int)smSrc[(BLOCK_N_X + 4) * (threadIdx.y + v) + threadIdx.x + u]) * mask[i][u][v];
//             }
//         }
//     }

//     val[0] = sqrt(val[0] * val[0] + val[1] * val[1]) / SCALE;
//     t[3 * (width * y + x) + z] = (val[0] > 255.0) ? 255 : val[0];
// }






__global__ void sobel(unsigned char *s, unsigned char *t, 
                                unsigned height, unsigned width, unsigned channels)
{

    int mask[MASK_N][MASK_X][MASK_Y] = {
    
        {{ -1, -4, -6, -4, -1},
        { -2, -8,-12, -8, -2},
        {  0,  0,  0,  0,  0},
        {  2,  8, 12,  8,  2},
        {  1,  4,  6,  4,  1}},

        {{ -1, -2,  0,  2,  1},
        { -4, -8,  0,  8,  4},
        { -6,-12,  0, 12,  6},
        { -4, -8,  0,  8,  4},
        { -1, -2,  0,  2,  1}}

    };

    int basex = blockIdx.x * blockDim.x;
    int basey = blockIdx.y * blockDim.y;
    int basez = blockIdx.z * blockDim.z;
    
    int nextBasex = basex + blockDim.x;
    int nextBasey = basey + blockDim.y;

    int x = basex + threadIdx.x;
    int y = basey + threadIdx.y;
    int z = basez + threadIdx.z;

    if(x >= width || y >= height)return;

    __shared__ unsigned char smSrc[(BLOCK_N_X + 4) * (BLOCK_N_Y + 4)];

    smSrc[(BLOCK_N_X + 4) * threadIdx.y + threadIdx.x] = s[channels * ((width + 4) * y + x) + z];

    if(threadIdx.x < 4){
        smSrc[(BLOCK_N_X + 4) * threadIdx.y + BLOCK_N_X + threadIdx.x] =\
                         s[channels * ((width + 4) * y + BLOCK_N_X + x) + z];
    }

    if(threadIdx.y < 4){
        smSrc[(BLOCK_N_X + 4) * (BLOCK_N_Y + threadIdx.y) + threadIdx.x] =\
                         s[channels * ((width + 4) * (BLOCK_N_Y + y) + x) + z];
    }

    if(threadIdx.x < 4 && threadIdx.y < 4){
        smSrc[(BLOCK_N_X + 4) * (BLOCK_N_Y + threadIdx.y) + BLOCK_N_X + threadIdx.x] =\
                         s[channels * ((width + 4) * (BLOCK_N_Y + y) + BLOCK_N_X + x) + z];
    }


    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){

        for(int dx=0; dx < BLOCK_N_X + 4; dx++) {
            for(int dy=0; dy < BLOCK_N_Y + 4; dy++) {

                if(smSrc[(BLOCK_N_X + 4) * dy + dx] !=\
                    s[channels * ((width + 4) * (basey + dy) + basex + dx) + z]){
                
                        printf("%d/%d, %d/%d, %d, %d\n", basex + dx, width, basey + dy, height,
                        smSrc[(BLOCK_N_X + 4) * dy + dx], 
                        s[channels * ((width + 4) * (basey + dy) + basex + dx) + z]);

                }

            }
        }

    }



    double val[2] = {0.0};

    for (int i = 0; i < MASK_N; ++i) {
        for (int v = 0; v <= 4; ++v) {     
            for (int u = 0; u <= 4; ++u) { 
                val[i] += smSrc[(BLOCK_N_X + 4) * (threadIdx.y + v) + (threadIdx.x + u)]\
                 * mask[i][u][v];
            }
        }
    }

    val[0] = sqrt(val[0]*val[0] + val[1]*val[1]) / SCALE;

    // if(y==619 && x == 800){
    //     printf("val[0]: %f\n", val[0]);
    // }

    // for(int srcx=0; srcx < width + 4; srcx++){
    //     for(int srcy=0; srcy < height + 4; srcy++){
    //         if((srcx >= 2) && (srcx <= width + 1) && (srcy >= 2) && (srcy <= width + 1))continue;

    //         for(int srcz=0;srcz<3;srcz++){
    //             if(!s[channels * ((width + 4) * srcy + srcx) + srcz]){
    //                 printf("!0: src_img[i]: %d\n", (int)s[channels * ((width + 4) * srcy + srcx) + srcz]);
    //             }
    //         }
    //     }        
    // }

    const unsigned char c = (val[0] > 255.0) ? 255 : val[0];



    t[channels * (width * y + x) + z] = c;

}





int main(int argc, char** argv) {
    assert(argc == 3);
    
    
    unsigned height, width, channels, gridNx, gridNy;
    unsigned char *src_img = NULL;
    read_png(argv[1], &src_img, &height, &width, &channels);
    assert(channels == 3);
    printf("width x height: %d x %d\n", width, height);
    // cc_check_param(width, height);


    gridNx = width / BLOCK_N_X + 1;
    gridNy = height / BLOCK_N_Y + 1;
    dim3 nThreadsPerBlock(BLOCK_N_X, BLOCK_N_Y, BLOCK_N_Z);
    dim3 nBlocks(gridNx, gridNy, GRID_N_Z);

    unsigned char *devSrc, *devDst;
    cudaMalloc(&devSrc, (height + 4) * (width + 4) * channels * sizeof(unsigned char));
    cudaMemcpy(devSrc, src_img, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // printf(" (height + 4) * (width + 4) * channels: %d\n",  (height + 4) * (width + 4) * channels);
    // for(int x=0; x < width + 4; x++){
    //     for(int y=0; y < height + 4; y++){
    //         if((x >= 2) && (x <= width + 1) && (y >= 2) && (y <= width + 1))continue;

    //         for(int z=0;z<3;z++){
    //             if(src_img[channels * ((width + 4) * y + x) + z]){
    //                 printf("!0: src_img[i]: %d\n", (int)src_img[channels * ((width + 4) * y + x) + z]);
    //             }
    //         }
    //     }        
    // }
    
    


    cudaMalloc(&devDst, height * width * channels * sizeof(unsigned char));

    sobel<<<nBlocks, nThreadsPerBlock>>>(devSrc, devDst, height, width, channels); 
    cudaDeviceSynchronize();
    

    unsigned char* dst_img =
        (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));
    cudaMemcpy(dst_img, devDst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst_img, height, width, channels);

    cudaFree(devSrc);
    cudaFree(devDst);
    free(src_img);
    free(dst_img);

    return 0;
}








// #include <stdio.h>

// __global__
// void saxpy(int n, unsigned char a, unsigned char *x, unsigned char *y)
// {
//     int i = blockIdx.x*blockDim.x + threadIdx.x;
//     if (i < n) y[i] = a*x[i] + y[i];
// }

// int main(void)
// {
//     int N = 1<<20;
//     unsigned char *x, *y, *d_x, *d_y;
//     x = (unsigned char*)malloc(N*sizeof(unsigned char));
//     y = (unsigned char*)malloc(N*sizeof(unsigned char));

//     cudaMalloc(&d_x, N*sizeof(unsigned char)); 
//     cudaMalloc(&d_y, N*sizeof(unsigned char));

//     for (int i = 0; i < N; i++) {
//         x[i] = 1.0f;
//         y[i] = 2.0f;
//     }

//     cudaMemcpy(d_x, x, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y, y, N*sizeof(unsigned char), cudaMemcpyHostToDevice);

//     // Perform SAXPY on 1M elements
//     saxpy<<<(N+255)/256, 256>>>(N, 2, d_x, d_y);

//     cudaMemcpy(y, d_y, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

//     // unsigned char maxError = 0.0f;
//     // for (int i = 0; i < N; i++)
//     //     maxError = max(maxError, abs(y[i]-4.0f));
//     // printf("Max error: %f\n", maxError);

//     cudaFree(d_x);
//     cudaFree(d_y);
//     free(x);
//     free(y);
// }
