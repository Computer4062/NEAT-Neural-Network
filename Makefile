CC = gcc
CFLAGS = -Wall -O2 -Iinclude
LDFLAGS = -lm

all: gen_model run_model

# Generate/train a network -> writes a .neat file
# Uses everything in src/ (whichever file holds main() for training) + neat.h
gen_model:
	$(CC) $(CFLAGS) src/*.c -o gen_model $(LDFLAGS)

# Load and run a saved .neat file -> standalone, no src/ files needed
run_model:
	$(CC) $(CFLAGS) examples/run_model.c -o run_model $(LDFLAGS)

clean:
	rm -f gen_model run_model

.PHONY: all clean gen_model run_model