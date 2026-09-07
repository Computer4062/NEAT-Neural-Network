# NEAT XOR

A from-scratch implementation of NEAT (NeuroEvolution of Augmenting
Topologies) in C, evolving a neural network that solves XOR.

## Directory structure

```
neat_xor/
├── Makefile
├── README.md
├── include/
│   ├── neat.h            # all structs (Genome, Species, Population) + declarations
│   └── neat_model.h      # standalone single-header loader/runner (see below)
├── src/
│   ├── genome.c           # genome creation, mutation, crossover, compatibility distance
│   ├── network.c          # feed-forward activation of a genome's phenotype network
│   ├── species.c           # speciation + fitness sharing
│   ├── population.c        # population lifecycle, selection, reproduction
│   ├── genome_io.c          # save/load a genome to the .neat file format
│   └── main.c                # XOR fitness function and the evolution loop
└── examples/
    └── run_model.c            # minimal demo: loads a .neat file using ONLY neat_model.h
```

## Build & run

```
make
./neat_xor
```

Each generation prints the number of species, the best fitness in the
current generation, and the best fitness seen so far. A perfect XOR score
is 16.0 (closeness = 4 - sum_squared_error, fitness = closeness^2); the run
stops as soon as a genome scores above 15.5, prints its evolved topology
and outputs on all four XOR cases, and **saves the trained network to
`xor.neat`** in the current directory.

## Reusing a trained model elsewhere (skip retraining)

Training produces `xor.neat`, a plain-text file describing the evolved
network's nodes, connections, and weights. To use it in a *different*
C program — on a different machine, with none of this project's training
code — you only need two things:

1. `include/neat_model.h` — a self-contained single-header library (no
   other file from this project required). Copy it into your own project.
2. The `.neat` file itself (e.g. `xor.neat`).

In your program:

```c
#define NEAT_MODEL_IMPLEMENTATION   /* exactly once, in one .c file */
#include "neat_model.h"

int main(void) {
    NeatModel m;
    if (!neat_model_load("xor.neat", &m)) {
        /* handle missing/corrupt file */
    }

    double inputs[3]  = {1, 0, 1};   /* size = m.num_inputs  */
    double outputs[1];                /* size = m.num_outputs */
    neat_model_run(&m, inputs, outputs);

    neat_model_free(&m);
    return 0;
}
```

Compile with `-lm` (it uses `exp()`). `examples/run_model.c` is exactly
this pattern — build and run it standalone with:

```
make demo
./examples/run_model xor.neat
```

This has been verified to work when only `neat_model.h`, `run_model.c`,
and the `.neat` file are copied into an otherwise-empty directory — it
reproduces the trained network's outputs exactly, with no dependency on
`neat.h` or any of the training/evolution sources.

### The `.neat` file format

Plain text, easy to inspect or hand-edit:

```
NEAT_MODEL 1
inputs 3
outputs 1
num_nodes 7
node 0 INPUT
node 1 INPUT
node 2 INPUT
node 3 OUTPUT
node 24 HIDDEN
...
num_conns 13
conn 0 3 -0.6809131411 1
conn 1 3 -0.9529228543 1
...
```

Each `conn` line is `in_node out_node weight enabled(0|1)`. Innovation
numbers aren't stored — they only matter during evolution/crossover, not
for running an already-trained network.

## How the algorithm works

- **Genome encoding**: a genome is a list of node genes (input / output /
  hidden) and connection genes (in-node, out-node, weight, enabled flag,
  historical *innovation number*).
- **Innovation numbers**: `Population` keeps a global table mapping
  `(in_node, out_node) -> innovation`, so identical structural mutations
  arising in different genomes get the same innovation number, which is
  what makes meaningful crossover between differently-structured genomes
  possible.
- **Mutation**:
  - weight mutation (perturb or reset),
  - add-connection (rejected if it would duplicate an existing connection
    or introduce a cycle, keeping networks strictly feed-forward),
  - add-node (splits an existing connection in two, preserving behavior).
- **Crossover**: matching genes (same innovation number) are inherited
  randomly from either parent; disjoint/excess genes are inherited from
  the fitter parent.
- **Speciation**: genomes are grouped by a compatibility-distance formula
  (excess genes, disjoint genes, average weight difference of matching
  genes). Fitness sharing means a species' reproductive budget is
  proportional to its *average* fitness, protecting topological
  innovations from being outcompeted before they've had a chance to
  optimize their weights.
- **Reproduction**: each generation, species reproduce in proportion to
  their share of total average fitness; healthy species (>=5 members) keep
  their champion unchanged (elitism); the rest are filled by crossover or
  cloning plus mutation.

## Tuning

Hyperparameters (population size, mutation rates, compatibility threshold,
etc.) live at the top of `include/neat.h` and `src/main.c` if you want to
experiment.
