CC       ?= gcc
CFLAGS   ?= -O3 -g -fno-inline -Wall -fopenmp
LDFLAGS  ?= -lm -fopenmp

TARGET    = toy_tc
SRCS      = toy_tc.c

all: $(TARGET) symbols

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

symbols: $(TARGET)
	@if [ -f ../../tracer/generate_symbols.py ]; then \
		python3 ../../tracer/generate_symbols.py $(TARGET) --funcs tc_order_by_degree,tc_intersect_neighbors,tc_intersect_chunk,run_tc; \
	fi

clean:
	rm -f $(TARGET) function_symbols.txt

.PHONY: all symbols clean
