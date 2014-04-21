#ifndef __PLOT_H__
#define __PLOT_H__

#include <GL/freeglut.h>

#ifdef OPENIL
#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>
#endif

void plot_init();
void plot_clean();

int *plot_render_particles(double *x, double radius, long N, double L,
        double *shade, int forces, double *v, int doarrows);
int plot_clear_screen();
int plot_exit_func();

void plot_init_opengl();
void plot_end_opengl();
void plot_set_draw_color(float r, float g, float b, float a);

#endif
