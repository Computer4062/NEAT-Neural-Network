/* Minimal standalone program: loads a trained model saved by neat_xor and
   runs it. This file only depends on neat_model.h -- none of the training
   sources (genome.c, population.c, etc.) are needed. This is the pattern
   to copy into a *different* program on a *different* computer: bring
   neat_model.h and the .neat file, nothing else.

   Build:  gcc -O2 -Iinclude examples/run_model.c -o run_model -lm
   Run:    ./run_model xor.neat
*/

#define NEAT_MODEL_IMPLEMENTATION
#include "neat_model.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "neural.neat";

    NeatModel m;
    if (!neat_model_load(path, &m)) {
        fprintf(stderr, "Could not load model from %s\n", path);
        return 1;
    }

    printf("Loaded %s: %d inputs, %d outputs, %d nodes, %d connections\n",
           path, m.num_inputs, m.num_outputs, m.num_nodes, m.num_conns);

    /* Assumes the classic XOR encoding this project trains with: two real
       inputs plus a bias input fixed at 1.0. Adjust for your own model's
       input layout. */
    double cases[4][3] = {
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1},
    };

    for (int i = 0; i < 4; i++) {
        double out[1];
        neat_model_run(&m, cases[i], out);
        printf("  %.0f XOR %.0f -> %.4f\n", cases[i][0], cases[i][1], out[0]);
    }

    neat_model_free(&m);
    return 0;
}
