CC       ?= gcc
CFLAGS   ?= -O3 -g -fno-inline -Wall -fopenmp
LDFLAGS  ?= -lm -fopenmp

TARGET    = toy_sssp
SRCS      = toy_sssp.c

all: $(TARGET) symbols

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

symbols: $(TARGET)
	@if [ -f ../../tracer/generate_symbols.py ]; then \
		python3 ../../tracer/generate_symbols.py $(TARGET) --funcs sssp_relax_edges,sssp_bucket_manage,run_sssp; \
	fi

clean:
	rm -f $(TARGET) function_symbols.txt

.PHONY: all symbols clean
