#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace std;

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



#define GRID_N_Z 1

#define BLOCK_N_X 32
#define BLOCK_N_Y 4
#define BLOCK_N_Z 3






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






__global__ void sobel(unsigned char *s, unsigned char *t, 
                                unsigned height, unsigned width, unsigned channels)
{

    short mask[MASK_N * MASK_X * MASK_Y] = {

          -1, -4, -6, -4, -1,
          -2, -8,-12, -8, -2,
           0,  0,  0,  0,  0,
           2,  8, 12,  8,  2,
           1,  4,  6,  4,  1,
         
          -1, -2,  0,  2,  1,
          -4, -8,  0,  8,  4,
          -6,-12,  0, 12,  6,
          -4, -8,  0,  8,  4,
          -1, -2,  0,  2,  1} ;

    const unsigned short tidx_z = threadIdx.x;
    const unsigned short tidx_x = threadIdx.y;
    const unsigned short tidx_y = threadIdx.z;
    const int bidx_z = blockIdx.x;
    const int bidx_x = blockIdx.y;
    const int bidx_y = blockIdx.z;
    const int bdim_z = blockDim.x;
    const int bdim_x = blockDim.y;
    const int bdim_y = blockDim.z;
    const int basez = bidx_z * bdim_z;
    const int basex = bidx_x * bdim_x;
    const int basey = bidx_y * bdim_y;
    const int z = basez + tidx_z;
    const int x = basex + tidx_x;
    const int y = basey + tidx_y; 

    int idx_raw, idx_divRound, y_group, y_id, char_batch_idx;

    unsigned long long idx_raw_64, idx_divRound_64;

    __shared__ unsigned char smSrc[128 * (BLOCK_N_Y + 4)];
    __shared__ unsigned short xzBase[BLOCK_N_Y + 4];

    // if(x > width + 4 - 1 || y > height + 4 - 1) return;
    


    if(tidx_x < 22 && tidx_z < 3 && tidx_y == 0 && 3 * tidx_x + tidx_z <= 63 ){ 
        
        y_group = (3 * tidx_x + tidx_z) / 32; 
        y_id =  ((3 * tidx_x + tidx_z) - y_group * 32 ) / 8;
        char_batch_idx = (3 * tidx_x + tidx_z) - y_group * 32 - y_id * 8;


        if((y_group * BLOCK_N_Y + basey + y_id) <= (height + 4 - 1)){
            idx_raw_64 = 3 * ((unsigned long long)(width + 4)) * ((unsigned long long)(y_group * BLOCK_N_Y + basey + y_id));
            idx_raw_64 += 3 * (unsigned long long)basex;
            idx_raw_64 += (unsigned long long)basez;

            idx_divRound_64 =  (idx_raw_64 >> 4);

            xzBase[y_group * BLOCK_N_Y + y_id] = idx_raw_64 - idx_divRound_64 * 16;
            idx_divRound_64 += char_batch_idx;

            reinterpret_cast<int4 *>(smSrc)[8 * (y_group * BLOCK_N_Y + y_id) +\
                                    char_batch_idx] = reinterpret_cast<int4 *>(s)[idx_divRound_64];
        }                                
    }


    if(x >= width || y >= height)return;

    __syncthreads();
   
    short val0 = 0;
    short val1 = 0;
    float result;
    unsigned char a;


    for (char u = 0; u <= 1; ++u) { 
        for (char v = 0; v <= 4; ++v) { 
            idx_raw = 128 * (tidx_y + v) + xzBase[tidx_y + v] + 3 * (tidx_x + u) + tidx_z;
            a = smSrc[idx_raw];
            val0 += a * mask[5 * u + v];
        }
    }

    for (char u = 3; u <= 4; ++u) { 
        for (char v = 0; v <= 4; ++v) { 
            idx_raw = 128 * (tidx_y + v) + xzBase[tidx_y + v] + 3 * (tidx_x + u) + tidx_z;
            a = smSrc[idx_raw];
            val0 += a * mask[5 * u + v];
        }
    }

    
    for (char u = 0; u <= 4; ++u) { 
        for (char v = 0; v <= 1; ++v) {     
            idx_raw = 128 * (tidx_y + v) + xzBase[tidx_y + v] + 3 * (tidx_x + u) + tidx_z;
            a = smSrc[idx_raw];
            val1 += a * mask[25 + 5 * u + v];
        }

        for (char v = 3; v <= 4; ++v) {     
            idx_raw = 128 * (tidx_y + v) + xzBase[tidx_y + v] + 3 * (tidx_x + u) + tidx_z;
            a = smSrc[idx_raw];
            val1 += a * mask[25 + 5 * u + v];
        }
    }


    result = sqrtf(((int)val0) * ((int)val0) + ((int)val1)*((int)val1)) / SCALE;
    const unsigned char c = (result > 255.0) ? 255 : result;

    t[3 * (width * y + x) + z] = c;

}


int main(int argc, char** argv) {
    assert(argc == 3);
    
    
    unsigned height, width, channels, gridNx, gridNy;
    unsigned char *src_img = NULL;
    unsigned char *devSrc, *devDst;
    

    auto start = high_resolution_clock::now();
    read_png(argv[1], &src_img, &height, &width, &channels);
    cudaMalloc(&devSrc, (height + 4) * (width + 4) * channels * sizeof(unsigned char));
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"read_png() time: "<<duration.count()<<" us"<<endl;


    assert(channels == 3);
    // printf("width x height: %d x %d\n", width, height);


    gridNx = width / BLOCK_N_X + 1;
    gridNy = height / BLOCK_N_Y + 1;
    dim3 nThreadsPerBlock(BLOCK_N_Z, BLOCK_N_X, BLOCK_N_Y);
    dim3 nBlocks(GRID_N_Z, gridNx, gridNy);

    
    

    start = high_resolution_clock::now();
    cudaMemcpy(devSrc, src_img, (height + 4) * (width + 4) * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout<<"cudaMemcpyHostToDevice time: "<<duration.count()<<" us"<<endl;


    cudaMalloc(&devDst, height * width * channels * sizeof(unsigned char));


    start = high_resolution_clock::now();
    sobel<<<nBlocks, nThreadsPerBlock>>>(devSrc, devDst, height, width, channels); 
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout<<"kernel time: "<<duration.count()<<" us"<<endl;

    unsigned char* dst_img =
        (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));
                

    start = high_resolution_clock::now();
    cudaMemcpy(dst_img, devDst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout<<"cudaMemcpyDeviceToHost time: "<<duration.count()<<" us"<<endl;



    start = high_resolution_clock::now();
    write_png(argv[2], dst_img, height, width, channels);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout<<"write_png() time: "<<duration.count()<<" us"<<endl;


    cudaFree(devSrc);
    cudaFree(devDst);
    // free(src_img);
    free(dst_img);

    return 0;
}


