CC = mpicc
CFLAGS = -Wall -O2
LDFLAGS = -lm

TARGET = psrs

SRCS = psrs.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TARGET)
	@echo "Testing PSRS output..."
	mpirun -np 3 ./psrs 100

clean:
	rm -f $(OBJS) $(TARGET)
