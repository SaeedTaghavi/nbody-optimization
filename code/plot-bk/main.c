//===================================================
// Author: Matthew Bierbaum
//===================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#ifdef PLOT
#include "plot.h"
#endif

#ifdef FPS
#include <time.h>
#endif

#define MAX(x,y) ((x>y)?(x):(y))
#define MIN(x,y) ((x<y)?(x):(y))
#define PI 3.141592653589
#define EPSILON DBL_EPSILON

void simulate(int seed);
void coords_to_index(double *x, int size, int *index, double L);

void ran_seed(long j);
double ran_ran2();
unsigned long long int vseed;
unsigned long long int vran;

//===================================================
// the main function
//===================================================
int main(int argc, char **argv){
    if (argc == 1) 
        simulate(0);
    else if (argc == 2)
        simulate(atoi(argv[1]));
    else {
        printf("usage:\n");
        printf("\t./entbody [seed]\n");
    }
    return 0;
}


//==================================================
// simulation
//==================================================
void simulate(int seed){
    ran_seed(seed);

    int i, j;
    int N = 1024;
    int NMAX = 50;
    double radius = 1.0;
    double L = 0.94*sqrt(PI*radius*radius*N);

    double epsilon = 25.0;
    double dt = 1e-1;
    double t = 0.0;
    double R = 2*radius; 
    double R2 = R*R;

    double *col = (double*)malloc(sizeof(double)*N); 
    double *x = (double*)malloc(sizeof(double)*2*N);
    double *v = (double*)malloc(sizeof(double)*2*N);
    double *f = (double*)malloc(sizeof(double)*2*N);
    for (i=0; i<2*N; i++){x[i] = v[i] = f[i] = 0.0;}

    #ifdef PLOT 
    double time_end = 1e20;
    #else
    double time_end = 1e1;
    #endif

    #ifdef PLOT 
        int *key;
        plot_init(); 
        plot_clear_screen();
        key = plot_render_particles(x, radius, N, L, col, 0, v, 1);
    #endif

    for (i=0; i<N; i++){
        x[2*i+0] = L*ran_ran2();
        x[2*i+1] = L*ran_ran2();
    }   

    //-------------------------------------------------------
    // make boxes for the neighborlist
    int size;
    int size_total;
    size = (int)(L / R); 
    size_total = size*size;

    int *count = (int*)malloc(sizeof(int)*size_total);
    int *cells = (int*)malloc(sizeof(int)*size_total*NMAX);
    for (i=0; i<size_total; i++) count[i] = 0;
    for (i=0; i<size_total*NMAX; i++) cells[i] = 0;

    //==========================================================
    // where the magic happens
    //==========================================================
    int frames = 0;
    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);

    for (t=0.0; t<time_end; t+=dt){

        int idx[2];
        for (i=0; i<size_total; i++)
            count[i] = 0;

        for (i=0; i<N; i++){
            coords_to_index(&x[2*i], size, idx, L);
            int t = idx[0] + idx[1]*size;
            cells[NMAX*t + count[t]] = i;
            count[t]++; 
        }

        int tx, ty;
        double dx, dy;
        int ind, n;
        double l, co, co1, dist;

        #ifdef OPENMP
        #pragma omp parallel for private(i,dx,idx,tt,goodcell,tix,ind,j,n,image,k,dist,r0,l,co,wlen,vlen,vhappy)
        #endif 
        for (i=0; i<N; i++){
            f[2*i+0] = 0.0;
            f[2*i+1] = 0.0;
            coords_to_index(&x[2*i], size, idx, L);

            for (tx=MAX(0,idx[0]-1); tx<=MIN(size-1,idx[0]+1); tx++){
            for (ty=MAX(0,idx[1]-1); ty<=MIN(size-1,idx[1]+1); ty++){
                ind = tx + ty*size; 

                for (j=0; j<count[ind]; j++){
                    n = cells[NMAX*ind+j];

                    dist = 0.0;
                    dx = x[2*n+0] - x[2*i+0];
                    dy = x[2*n+1] - x[2*i+1];
                    dist = dx*dx + dy*dy;

                    //===============================================
                    // force calculation - hertz
                    if (dist > EPSILON && dist < R2){
                        l = sqrt(dist);
                        co1 = (1-l/R);
                        co = epsilon * co1*sqrt(co1) * (l<R);
                        
                        f[2*i+0] += -dx/l * co;
                        f[2*i+1] += -dy/l * co;
                        col[i] += co*co;
                    }
                }
            } } 

            f[2*i+0] -= 0.5*v[2*i+0];
            f[2*i+1] -= 0.5*v[2*i+1];
        }
        #ifdef OPENMP
        #pragma omp barrier
        #endif

        // now integrate the forces since we have found them
        #ifdef OPENMP
        #pragma omp parallel for private(j)
        #endif 
        for (i=0; i<N;i++){
            // Newton-Stomer-Verlet
            v[2*i+0] += f[2*i+0] * dt;
            v[2*i+1] += f[2*i+1] * dt;

            x[2*i+0] += v[2*i+0] * dt;
            x[2*i+1] += v[2*i+1] * dt;

            // boundary conditions 
            for (j=0; j<2; j++){
                const double rst = 0.1;
                if (x[2*i+j] >= L){x[2*i+j] = 2*L-x[2*i+j]; v[2*i+j] *= -rst;}
                if (x[2*i+j] < 0) {x[2*i+j] = -x[2*i+j];    v[2*i+j] *= -rst;}
            }

            col[i] = col[i]/12; 
        }
        #ifdef OPENMP
        #pragma omp barrier
        #endif

        #ifdef PLOT 
        int skip = 10;
        int start = 20;
        if (frames % skip == 0 && frames >= start){
            plot_clear_screen();
            key = plot_render_particles(x, radius, N, L,col, 1, v, 1);
            if (key['q'] == 1)
                break;
           
        }
        #endif
        frames++;
    }
    // end of the magic, cleanup
    //----------------------------------------------
    #ifdef FPS
    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &end);
    printf("fps = %f\n", frames/((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9));
    #endif

    free(cells);
    free(count);
 
    free(x);
    free(v);
    free(f);
    free(col);

    #ifdef PLOT
    plot_clean(); 
    #endif
}

//=================================================
// extra stuff
//=================================================
void ran_seed(long j){
  vseed = j;  vran = 4101842887655102017LL;
  vran ^= vseed; 
  vran ^= vran >> 21; vran ^= vran << 35; vran ^= vran >> 4;
  vran = vran * 2685821657736338717LL;
}

double ran_ran2(){
    vran ^= vran >> 21; vran ^= vran << 35; vran ^= vran >> 4;
    unsigned long long int t = vran * 2685821657736338717LL;
    return 5.42101086242752217e-20*t;
}

inline void coords_to_index(double *x, int size, int *index, double L){   
    index[0] = (int)(x[0]/L  * size);
    index[1] = (int)(x[1]/L  * size);
}

