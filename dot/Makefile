CC := nvcc
CFLAGS := -std=c++11

all: dot

dot: dot.cu
	$(CC) $(CFLAGS) -o $@ $<

test: dot
	./dot 128 256
	./dot 100 256
	./dot 256 128
	./dot 200 128
	./dot 64 256
	./dot 512 64
	./dot 32 32
	./dot 1024 32
	./dot 16 2048
	./dot 2048 16

.PHONY: clean
clean:
	rm -f dot