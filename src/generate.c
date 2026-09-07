#include "neat.h"
#include <stdio.h>
#include <time.h>

/* Squared "closeness" score: perfect play scores 16.0 (= 4^2). Squaring
   rewards genomes that are close to correct on all four cases, which gives
   evolution a smoother gradient to climb than raw sum-squared-error. */
static double xor_fitness(const Genome *g, const int num_samples, const int num_inputs, const int num_outputs, 
							double inputs[num_samples][num_inputs], double expectations[num_samples]) {
    double error = 0;
    for (int i = 0; i < num_samples; i++) {
        double out[1];
        network_evaluate(g, expectations[i], out, num_inputs, num_outputs);
        double diff = expectations[i] - out[0];
        error += diff * diff;
    }
    double closeness = 4.0 - error;
    if (closeness < 0) closeness = 0;
    return closeness * closeness;
}

static void print_genome_report(const Genome *g, const int num_samples, const int num_inputs, const int num_outputs,
							double inputs[num_samples][num_inputs], double expectations[num_samples]) {
    int enabled = 0;
    for (int i = 0; i < g->num_conns; i++) if (g->conns[i].enabled) enabled++;
    printf("  nodes: %d | connections: %d (enabled: %d)\n", g->num_nodes, g->num_conns, enabled);
    for (int i = 0; i < num_samples; i++) {
        double out[1];
        network_evaluate(g, inputs[i], out, num_samples, num_outputs);
        printf("  %.0f XOR %.0f -> %.4f  (expected %.0f)\n",
               inputs[i][0], inputs[i][1], out[0], expectations[i]);
    }
}

void generate_neural_net(const int pop_size, const int num_inputs, const int num_samples, const int num_outputs, const int max_gens,
						double inputs[num_samples][num_inputs], double expectations[num_samples]) {
    srand((unsigned)time(NULL));

    Population pop = population_create(pop_size, num_inputs, num_outputs);
    speciate_population(&pop);

    int solved = 0;
    for (int gen = 0; gen < max_gens; gen++) {
        population_evaluate(&pop, xor_fitness);
        Genome *best = population_best(&pop);

        printf("Gen %3d | species: %2d | best fitness: %7.4f | best-ever: %7.4f\n",
               gen, pop.num_species, best->fitness, pop.best_ever_fitness);

        if (best->fitness > 15.5) { /* max possible is 16.0 */
            printf("\nSolved XOR at generation %d!\n", gen);
            print_genome_report(best);
            if (genome_save(best, num_inputs, num_outputs, "xor.neat"))
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
               max_gens);
        print_genome_report(&pop.best_ever);
        if (genome_save(&pop.best_ever, num_inputs, num_outputs, "neural.neat"))
            printf("\nSaved best-so-far model to xor.neat\n");
    }

    population_free(&pop);
    return 0;
}
