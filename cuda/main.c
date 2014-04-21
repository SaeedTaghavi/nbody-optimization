#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#ifdef PLOT
#include "plot.h"
#endif

void simulate(int s);

#define NPOW        (10+6)
#define TPOW        (6 + NPOW%2)
#define N           (1 << NPOW)
#define NTHREADS    (1 << TPOW)
#define NBLOCKS     (1 << ((NPOW-TPOW)/2))

#define MAX(x,y) ((x>y)?(x):(y))
#define MIN(x,y) ((x<y)?(x):(y))
#define NMAX 50
#define pi 3.141592653589
#define EPSILON FLT_EPSILON

typedef unsigned long long int ullong;

// random number generator functions
void ran_seed(long j);
float ran_ran2();

#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

#define CUDA_SAFE_CALL( call) {                                            \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define CUT_DEVICE_INIT(dev) {                                               \
    int deviceCount;                                                         \
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));                        \
    if (deviceCount == 0) {                                                  \
        fprintf(stderr, "cutil error: no devices supporting CUDA.\n");       \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    if (dev < 0) dev = 0;                                                    \
    if (dev > deviceCount-1) dev = deviceCount - 1;                          \
    cudaDeviceProp deviceProp;                                               \
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));               \
    if (deviceProp.major < 1) {                                              \
        fprintf(stderr, "cutil error: device does not support CUDA.\n");     \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    CUDA_SAFE_CALL(cudaSetDevice(dev));                                      \
}

__device__ float mymod(float a, float b){
  return a - b*(int)(a/b) + b*(a<0);
}
//===================================================
// the main function
//===================================================
int main(int argc, char **argv){
    int seed_in = 0;

    int device = 0;
    CUT_DEVICE_INIT(device);

    if (argc == 1)
        simulate(seed_in);
    else if (argc == 2)
        simulate(atoi(argv[1]));//seed_in);
    else {
        printf("usage:\n");
        printf("\t./entbody [seed]\n");
    }
    return 0;
}

__global__ void nbl_reset(int *cells, int *count, int size_total){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;
    int i = idx + idy*NBLOCKS*NTHREADS;

    if (i < size_total) count[i] = 0;
    if (i < size_total*NMAX) cells[i] = 0;
}

__global__ void nbl_build(float *x, int *cells, int *count, int size,
        int size_total, float L){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;
    int i = idx + idy*NBLOCKS*NTHREADS;
    volatile int indx, indy;

    indx = __float2int_rz(x[2*i+0]/L*size);
    indy = __float2int_rz(x[2*i+1]/L*size);
    volatile int t  = indx + indy*size;
    volatile unsigned int ct = atomicAdd(&count[t], 1);
    volatile unsigned int bt = NMAX*t + ct;
    cells[bt] = i;
}

__global__ void makecopy(float *x, float *copyx){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;
    int i = idx + idy*NBLOCKS*NTHREADS;
    copyx[2*i+0] = x[2*i+0];
    copyx[2*i+1] = x[2*i+1];
}

//==================================================================
// the timestep - can be CPU or CUDA!
//==================================================================
__global__
void step(float *x, float *copyx, float *v, float radius, float *col,
          int *cells, int *count, int size, int size_total,
          float L, float R, float dt, float Tglobal){

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;
    int i = idx + idy*NBLOCKS*NTHREADS;
    int j;

    //==========================================
    // this is mainly for CUDA optimization
    int tx, ty;
    float dx, dy;
    int ind, tn;
    float dist;

    float px, py;
    float vx, vy;
    float fx, fy;

    float tcol;
    float R2 = R*R;

    //==========================================
    // find forces on all particles
    tcol  = col[i];
    px = copyx[2*i+0];
    py = copyx[2*i+1];
    vx = v[2*i+0];
    vy = v[2*i+1];

    fx = 0.0; fy = 0.0;

    int indx, indy;
    indx = (int)(px/L * size);
    indy = (int)(py/L * size);

    for (tx=MAX(0,indx-1); tx<=MIN(size-1,indx+1); tx++){
    for (ty=MAX(0,indy-1); ty<=MIN(size-1,indy+1); ty++){
        ind = tx + ty*size;
        for (j=0; j<count[ind]; j++){
            tn = cells[NMAX*ind+j];
            float px2 = copyx[2*tn+0];
            float py2 = copyx[2*tn+1];

            dist = 0.0;
            dx = px2 - px;
            dy = py2 - py;
            dist = dx*dx + dy*dy;

            //===============================================
            // force calculation
            if (dist > EPSILON && dist < R2){
                float r0 = 2*radius;
                float l = sqrt(dist);
                float co = 25 * (1-l/r0)*(1-l/r0) * (l<r0);
                fx += -co * dx;
                fy += -co * dy;
                tcol += co*co;
            }
        }
    } }

    fx -= vx;
    fy -= vy;

    vx += fx * dt;
    vy += fy * dt;
    px += vx * dt;
    py += vy * dt;

    //======================================
    // boundary conditions
    const float restoration = 0.1;
    if (px >= L){px = 2*L-px; vx *= -restoration;}
    if (px < 0) {px = -px;    vx *= -restoration;}
    if (px >= L-EPSILON || px < 0){px = mymod(px, L);}

    if (py >= L){py = 2*L-py; vy *= -restoration;}
    if (py < 0) {py = -py;    vy *= -restoration;}
    if (py >= L-EPSILON || py < 0){py = mymod(py, L);}

    tcol = tcol/4;

    col[i] = tcol;
    x[2*i+0] = px;  x[2*i+1] = py;
    v[2*i+0] = vx;  v[2*i+1] = vy;
}


//==================================================
// simulation
//==================================================
void simulate(int seed){
    ran_seed(seed);
#ifdef CUDA_NO_SM_11_ATOMIC_INTRINSICS
        printf("WARNING! Not using atomics!\n");
#endif
    float L = 0.0;
    float dt = 1e-1;
    float t = 0.0;
    float Tglobal = 0.0;
    float radius = 1.0;
    float R = 2*radius;

    int i;
    int mem_size_f = sizeof(float)*N;
    float *col   = (float*)malloc(mem_size_f);
    for (i=0; i<N; i++){ col[i] = 0.0;}

    float *x     = (float*)malloc(2*mem_size_f);
    float *v     = (float*)malloc(2*mem_size_f);
    float *copyx = (float*)malloc(2*mem_size_f);
    for (i=0; i<2*N; i++){x[i] = v[i] = copyx[i] = 0.0;}

    float time_end = 2e10;

    #ifdef PLOT
        int *key;
        plot_init(800); // 2**18 - 450, 2**20 - 900, 2**21 - 1100
        plot_clear_screen();
        key = plot_render_particles(x, radius, N, L,col);
    #endif

    L = 0.94*sqrt(pi*radius*radius*N);
    for (i=0; i<2*N; i++){
        x[i] = L*ran_ran2();
        v[i] = 0.0;
     }

    // make boxes for the neighborlist
    int size = (int)(L/R);
    int size_total = size*size;
    int *count = (int*)malloc(sizeof(int)*size_total);
    int *cells = (int*)malloc(sizeof(int)*size_total*NMAX);
    for (i=0; i<size_total; i++) count[i] = 0;
    for (i=0; i<size_total*NMAX; i++) cells[i] = 0;

    //==========================================================
    // where the magic happens
    //==========================================================
    int fmem_size = sizeof(float)*N;
    int fmem_siz2 = sizeof(float)*N*2;
    int mem_cell  = sizeof(int)*size_total;
    int mem_cell2 = sizeof(int)*size_total*NMAX;

    int *cu_count  = NULL;
    int *cu_cells  = NULL;
    float *cu_col  = NULL;
    float *cu_x    = NULL;
    float *cu_v    = NULL;
    float *cu_copyx= NULL;

    cudaMalloc((void**) &cu_count, mem_cell);
    cudaMalloc((void**) &cu_cells, mem_cell2);

    cudaMalloc((void**) &cu_col,   fmem_size);
    cudaMalloc((void**) &cu_x,     fmem_siz2);
    cudaMalloc((void**) &cu_v,     fmem_siz2);
    cudaMalloc((void**) &cu_copyx, fmem_siz2);

    printf("Copying problem...\n");
    cudaMemcpy(cu_col,   col,   fmem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_x,     x,     fmem_siz2, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_v,     v,     fmem_siz2, cudaMemcpyHostToDevice);
    cudaMemset(cu_count, 0, mem_cell);
    cudaMemset(cu_cells, 0, mem_cell2);
    ERROR_CHECK

    int frames = 0;

    float rate;
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    dim3 grid(NBLOCKS, NBLOCKS);
    dim3 block(NTHREADS, 1);

    for (t=0.0; t<time_end; t+=dt){
        nbl_reset<<<grid, block>>>(cu_cells, cu_count, size_total);
        nbl_build<<<grid, block>>>(cu_x, cu_cells, cu_count, size, size_total, L);
        makecopy<<<grid, block>>>(cu_x, cu_copyx);

        step<<<grid, block>>>(cu_x, cu_copyx, cu_v, radius, cu_col,
                    cu_cells, cu_count, size, size_total,
                    L, R, dt, Tglobal);
        cudaThreadSynchronize();
        ERROR_CHECK

        if (frames % 50 == 0){
            clock_gettime(CLOCK_REALTIME, &end);
            rate = frames/((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)/1e9);
            printf("N = %i\tframe = %i\trate = %f\r", N, frames, rate);
            fflush(stdout);
        }

        #ifdef PLOT
        if (frames % 1 == 0){
            cudaMemcpy(x, cu_x, fmem_siz2, cudaMemcpyDeviceToHost);
            cudaMemcpy(col, cu_col, fmem_size, cudaMemcpyDeviceToHost);

            plot_clear_screen();
            key = plot_render_particles(x, radius, N, L, col);
            if (key['q'] == 1) break;
        }
        #endif
        frames++;
    }

    clock_gettime(CLOCK_REALTIME, &end);
    rate = frames/((end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)/1e9);
    printf("N = %i\tframe = %i\trate = %f\n", N, frames, rate);

    free(cells);
    free(count);

    free(copyx);
    free(x);
    free(v);
    free(col);

    cudaFree(cu_count);
    cudaFree(cu_cells);
    cudaFree(cu_col);
    cudaFree(cu_x);
    cudaFree(cu_v);
    cudaFree(cu_copyx);
    ERROR_CHECK

    #ifdef PLOT
    plot_clean();
    #endif
}


ullong vseed;
ullong vran;

//=================================================
// random number generator 
//=================================================
void ran_seed(long j){
  vseed = j;  vran = 4101842887655102017LL;
  vran ^= vseed; 
  vran ^= vran >> 21; vran ^= vran << 35; vran ^= vran >> 4;
  vran = vran * 2685821657736338717LL;
}

float ran_ran2(){
    vran ^= vran >> 21; vran ^= vran << 35; vran ^= vran >> 4;
    ullong t = vran * 2685821657736338717LL;
    return 5.42101086242752217e-20*t;
}

