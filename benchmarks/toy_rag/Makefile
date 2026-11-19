# toy_rag Makefile
#
# Builds optimized benchmark with debug symbols for Pin tracing & ChampSim tracking.
#   make                   -> builds toy_rag and gen_rag_db, extracts symbols
#   make gen_rag_db        -> builds database generator
#   make db SCALE=16       -> generates rag_db_s16_d32_v64.bin
#   make symbols           -> extracts symbol offsets for ChampSim SymbolTracker
#   make clean

CC       := gcc
CFLAGS   := -O2 -g -Wall -Wextra
LDFLAGS  := -lm
TARGETS  := toy_rag gen_rag_db

SCALE    ?= 16
DEG      ?= 32
DIM      ?= 64
SEED     ?= 42
DB_BIN   ?= rag_db_s$(SCALE)_d$(DEG)_v$(DIM).bin

.PHONY: all clean symbols db

all: $(TARGETS) symbols

toy_rag: toy_rag.c toy_rag.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

gen_rag_db: gen_rag_db.c toy_rag.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

db: gen_rag_db
	./gen_rag_db $(DB_BIN) $(SCALE) $(DEG) $(DIM) $(SEED)

symbols: toy_rag
	@python3 ../../tracer/generate_symbols.py toy_rag \
		-f rag_encode_query,rag_graph_retrieval,rag_llm_generation \
		-c ../../NUMA_Csim ../../tracer/TRACES/TOY_RAG

clean:
	rm -f $(TARGETS) *.bin function_symbols.txt

