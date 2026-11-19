CC       ?= gcc
CFLAGS   ?= -O3 -g -fno-inline -Wall -fopenmp
LDFLAGS  ?= -lm -fopenmp

TARGET    = toy_cc
SRCS      = toy_cc.c

all: $(TARGET) symbols

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

symbols: $(TARGET)
	@if [ -f ../../tracer/generate_symbols.py ]; then \
		python3 ../../tracer/generate_symbols.py $(TARGET) --funcs cc_link_step,cc_compress_step,cc_sample_frequent,run_cc; \
	fi

clean:
	rm -f $(TARGET) function_symbols.txt

.PHONY: all symbols clean
