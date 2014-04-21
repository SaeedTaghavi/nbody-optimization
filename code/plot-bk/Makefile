# change this to 1 or 0 depending if you want a pretty plot,
# not sure what you want otherwise
DOPLOT = 1
FPS    = 1
OPENMP = 0

# standard compile options for the c++ executable
GCC = gcc
EXE = entbody
OBJS =  main.c
FLAGS = -Ofast -Wall -Werror
LIBFLAGS = -lm

ifeq ($(DOPLOT), 1)
    OBJS += plot.c
    FLAGS += -DPLOT
    LIBFLAGS += -lGL -lGLU -lglut 
endif

ifeq ($(OPENMP), 1)
    FLAGS += -fopenmp
    FLAGS += -DOPENMP
endif

ifeq ($(FPS),1)
    LIBFLAGS += -lrt
    FLAGS += -DFPS
endif

.PHONY: clean tidy

# default super-target
all: $(EXE)

# the standard executable
$(EXE): $(OBJS)
	$(GCC) $(FLAGS) $^ -o $@ $(LIBFLAGS)

tidy:
	@find | egrep "#" | xargs rm -f
	@find | egrep "\~" | xargs rm -f
	@find | egrep ".txt" | xargs rm -f

clean: $(EXE)
	rm -f $(EXE)
