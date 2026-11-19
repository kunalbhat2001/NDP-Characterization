CC ?= gcc
CFLAGS ?= -O3 -fopenmp -mavx2 -Wall -Wextra -std=gnu11
TARGET = toy_cxl_stream

all: $(TARGET)

$(TARGET): toy_cxl_stream.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
