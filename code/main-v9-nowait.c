//===================================================
// Author: Matthew Bierbaum
//===================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define IDX(i,j) (2*i+j)
//#define IDX(i,j) (i+N*j)
#define MAX(x,y) ((x>y)?(x):(y))
#define MIN(x,y) ((x<y)?(x):(y))
#define BOX(x, L, size) ((int)(x/L*size))
#define PI 3.141592653589
#define EPSILON DBL_EPSILON

void simulate(int seed);
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
    int N = 2048;
    int NMAX = 50;
    double radius = 1.0;
    double L = 0.94*sqrt(PI*radius*radius*N);

    double epsilon = 25.0;
    double dt = 1e-1;
    double t = 0.0;
    double R = 2*radius; 
    double R2 = R*R;
    double time_end = 5e2;

    double *col = (double*)malloc(sizeof(double)*N); 
    double *x = (double*)malloc(sizeof(double)*2*N);
    double *v = (double*)malloc(sizeof(double)*2*N);
    double *f = (double*)malloc(sizeof(double)*2*N);
    for (i=0; i<2*N; i++){x[i] = v[i] = f[i] = 0.0;}
    for (i=0; i<2*N; i++){x[i] = L*ran_ran2();}


    //-------------------------------------------------------
    // make boxes for the neighborlist
    int size = (int)(L/R);
    int size_total = size*size;
    int *count = (int*)malloc(sizeof(int)*size_total);
    int *cells = (int*)malloc(sizeof(int)*size_total*NMAX);
    for (i=0; i<size_total; i++) count[i] = 0;
    for (i=0; i<size_total*NMAX; i++) cells[i] = 0;

    int frames = 0;
    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);

    for (t=0.0; t<time_end; t+=dt){

        int idx, idy;
        for (i=0; i<size_total; i++) count[i] = 0;

        for (i=0; i<N; i++){
            idx = BOX(x[IDX(i,0)], L, size);
            idy = BOX(x[IDX(i,1)], L, size);

            int t = idx + idy*size;
            cells[NMAX*t + count[t]] = i;
            count[t]++; 
        }

        int tx, ty;
        double dx, dy;
        int ind, n;
        double l, co, co1, dist;

        #pragma omp parallel
        {
        #pragma omp for nowait schedule(dynamic,N/16) private(i,j,dx,dy,idx,idy,tx,ty,n,dist,l,co,co1,ind)
        for (i=0; i<N; i++){
            f[IDX(i,0)] = 0.0;
            f[IDX(i,1)] = 0.0;
            idx = BOX(x[IDX(i,0)], L, size);
            idy = BOX(x[IDX(i,1)], L, size);

            for (tx=MAX(0,idx-1); tx<=MIN(size-1,idx+1); tx++){
            for (ty=MAX(0,idy-1); ty<=MIN(size-1,idy+1); ty++){
                ind = tx + ty*size; 

                for (j=0; j<count[ind]; j++){
                    n = cells[NMAX*ind+j];

                    dist = 0.0;
                    dx = x[IDX(n,0)] - x[IDX(i,0)];
                    dy = x[IDX(n,1)] - x[IDX(i,1)];
                    dist = dx*dx + dy*dy;

                    //===============================================
                    // force calculation - hertz
                    if (dist > EPSILON && dist < R2){
                        l = sqrt(dist);
                        co1 = (1-l/R);
                        co = epsilon * co1*sqrt(co1) * (l<R);
                        
                        f[IDX(i,0)] -= co*dx/l;
                        f[IDX(i,1)] -= co*dy/l;
                        col[i] += co*co;
                    }
                }
            } } 

            f[IDX(i,0)] -= v[IDX(i,0)];
            f[IDX(i,1)] -= v[IDX(i,1)];
        } }

        #pragma omp parallel
        {
        #pragma omp for nowait schedule(dynamic,N/8)
        for (i=0; i<N;i++){
            // Newton-Stomer-Verlet
            v[IDX(i,0)] += f[IDX(i,0)] * dt;
            v[IDX(i,1)] += f[IDX(i,1)] * dt;

            x[IDX(i,0)] += v[IDX(i,0)] * dt;
            x[IDX(i,1)] += v[IDX(i,1)] * dt;

            // boundary conditions 
            const double rst = 0.1;
            if (x[IDX(i,0)] >= L){x[IDX(i,0)] = 2*L-x[IDX(i,0)]; v[IDX(i,0)] *= -rst;}
            if (x[IDX(i,0)] < 0) {x[IDX(i,0)] =    -x[IDX(i,0)]; v[IDX(i,0)] *= -rst;}
            if (x[IDX(i,1)] >= L){x[IDX(i,1)] = 2*L-x[IDX(i,1)]; v[IDX(i,1)] *= -rst;}
            if (x[IDX(i,1)] < 0) {x[IDX(i,1)] =    -x[IDX(i,1)]; v[IDX(i,1)] *= -rst;}

            col[i] = col[i]/12; 
        }
        }

        frames++;
    }

    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &end);
    printf("fps = %f\n", frames/((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9));

    free(cells);
    free(count);
 
    free(x);
    free(v);
    free(f);
    free(col);
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

