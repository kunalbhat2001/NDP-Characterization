CC       ?= gcc
CFLAGS   ?= -O3 -g -fno-inline -Wall -fopenmp
LDFLAGS  ?= -lm -fopenmp

TARGET    = toy_pr
SRCS      = toy_pr.c

all: $(TARGET) symbols

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

symbols: $(TARGET)
	@if [ -f ../../tracer/generate_symbols.py ]; then \
		python3 ../../tracer/generate_symbols.py $(TARGET) --funcs pr_pull_step,pr_pull_range,pr_dampen_and_norm,run_pagerank; \
	fi

clean:
	rm -f $(TARGET) function_symbols.txt

.PHONY: all symbols clean
