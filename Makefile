CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Iinclude
LDFLAGS = -lm

SRC = src/genome.c src/network.c src/species.c src/population.c src/genome_io.c src/main.c
OBJ = $(SRC:.c=.o)
BIN = neat_xor
DEMO = examples/run_model

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Standalone demo: only needs neat_model.h, proves it works with none of
# the training source files.
demo: $(DEMO)

$(DEMO): examples/run_model.c include/neat_model.h
	$(CC) $(CFLAGS) -o $(DEMO) examples/run_model.c $(LDFLAGS)

clean:
	rm -f $(OBJ) $(BIN) $(DEMO)

.PHONY: all demo clean
