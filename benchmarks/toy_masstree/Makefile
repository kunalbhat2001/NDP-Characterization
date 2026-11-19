CC       ?= gcc
CFLAGS   ?= -O3 -g -fno-inline -Wall -fopenmp
LDFLAGS  ?= -lm -fopenmp

TARGET    = toy_masstree
SRCS      = toy_masstree.c

all: $(TARGET) symbols

$(TARGET): $(SRCS) toy_masstree.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

symbols: $(TARGET)
	@if [ -f ../../tracer/generate_symbols.py ]; then \
		python3 ../../tracer/generate_symbols.py $(TARGET) --funcs masstree_encode_hash_key,masstree_trie_traverse,masstree_leaf_search,masstree_range_scan,run_masstree; \
	fi

clean:
	rm -f $(TARGET) function_symbols.txt

.PHONY: all symbols clean
