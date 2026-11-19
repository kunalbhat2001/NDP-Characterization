# toy_bfs Makefile
#
# Builds single-threaded, optimized, with debug symbols for PIN.
#   make                   → builds toy_bfs and gen_graph
#   make gen_graph         → builds graph generator
#   make graph SCALE=16    → generates graph_s16_d16.bin
#   make clean

CC       := gcc
CFLAGS   := -O2 -g -Wall -Wextra -fopenmp
TARGETS  := toy_bfs gen_graph

SCALE    ?= 16
DEG      ?= 16
SEED     ?= 42
GRAPH_BIN ?= graph_s$(SCALE)_d$(DEG).bin

.PHONY: all clean symbols graph

all: $(TARGETS) symbols

toy_bfs: toy_bfs.c toy_graph.h
	$(CC) $(CFLAGS) -o $@ $<

gen_graph: gen_graph.c toy_graph.h
	$(CC) $(CFLAGS) -o $@ $<

graph: gen_graph
	./gen_graph $(GRAPH_BIN) $(SCALE) $(DEG) $(SEED)

symbols: toy_bfs
	@python3 ../../tracer/generate_symbols.py toy_bfs -f bfs,bu_step,td_step,queue_to_bitmap,bitmap_to_queue,bfs_should_switch_bu -c ../../NUMA_Csim ../../experiments/toy_bfs_1C1S ../../tracer/TRACES/TOY_BFS

clean:
	rm -f $(TARGETS) *.bin
