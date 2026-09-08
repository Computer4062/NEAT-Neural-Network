#include "neat.h"
#include <stdio.h>
#include <time.h>

#define MAX_GENERATIONS 300
#define POP_SIZE        150

/* Squared "closeness" score: perfect play scores 16.0 (= 4^2). Squaring
   rewards genomes that are close to correct on all four cases, which gives
   evolution a smoother gradient to climb than raw sum-squared-error. */
static double xor_fitness(const Genome *g, void *user_data) {
    Dataset *data = (Dataset *)user_data;
    double error = 0;
    for (int i = 0; i < data->num_samples; i++) {
        double out[1];
        network_evaluate(g, data->expectations[i], out, data->num_inputs, data->num_outputs);
        double diff = data->expectations[i] - out[0];
        error += diff * diff;
    }
    double closeness = 4.0 - error;
    if (closeness < 0) closeness = 0;
    return closeness * closeness;
}

static void print_genome_report(const Genome *g, void *user_data) {
    Dataset *data = (Dataset *)user_data;
    int enabled = 0;
    for (int i = 0; i < g->num_conns; i++) if (g->conns[i].enabled) enabled++;
    printf("  nodes: %d | connections: %d (enabled: %d)\n", g->num_nodes, g->num_conns, enabled);
    for (int i = 0; i < data->num_samples; i++) {
        double out[1];
        network_evaluate(g, data->inputs[i], out, data->num_samples, data->num_outputs);
        printf("  %.0f XOR %.0f -> %.4f  (expected %.0f)\n",
               data->inputs[i][0], data->inputs[i][1], out[0], data->expectations[i]);
    }
}

void generate_neural_net(void* user_data) {
    Dataset *data = (Dataset*)user_data;
    srand((unsigned)time(NULL));

    Population pop = population_create(POP_SIZE, data->num_inputs, data->num_outputs);
    speciate_population(&pop);

    int solved = 0;
    for (int gen = 0; gen < MAX_GENERATIONS; gen++) {
        population_evaluate(&pop, xor_fitness);
        Genome *best = population_best(&pop);

        printf("Gen %3d | species: %2d | best fitness: %7.4f | best-ever: %7.4f\n",
               gen, pop.num_species, best->fitness, pop.best_ever_fitness);

        if (best->fitness > 15.5) { /* max possible is 16.0 */
            printf("\nSolved XOR at generation %d!\n", gen);
            print_genome_report(best, &data);
            if (genome_save(best, data->num_inputs, data->num_outputs, "xor.neat"))
                printf("\nSaved trained model to xor.neat\n");
            else
                fprintf(stderr, "\nWarning: failed to write xor.neat\n");
            solved = 1;
            break;
        }
        population_reproduce(&pop);
    }

    if (!solved) {
        printf("\nDid not fully converge within %d generations. Best genome found:\n",
               MAX_GENERATIONS);
        print_genome_report(&pop.best_ever, &data);
        if (genome_save(&pop.best_ever, data->num_inputs, data->num_outputs, "neural.neat"))
            printf("\nSaved best-so-far model to neat.neat\n");
    }

    population_free(&pop);
    return 0;
}
