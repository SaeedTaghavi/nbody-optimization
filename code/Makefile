OPENMP = 0

# standard compile options for the c++ executable
GCC = gcc
FLAGS = -O0 -Wall -Werror -fopenmp
LIBFLAGS = -lm -lrt

ifeq ($(OPENMP), 1)
    FLAGS += -fopenmp
    FLAGS += -DOPENMP
endif

.PHONY: clean tidy

all: $(patsubst %.c, %.exe, $(wildcard *.c))

%.exe: %.c Makefile
	$(GCC) $(FLAGS) $< -o $@ $(LIBFLAGS)

tidy:
	@find | egrep "#" | xargs rm -f
	@find | egrep "\~" | xargs rm -f
	@find | egrep ".txt" | xargs rm -f

clean: $(EXE)
	rm -f *.exe
