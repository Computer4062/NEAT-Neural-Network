#include "neat.h"
#include <string.h>
#include <math.h>

static double rd(void) { return (double)rand() / (double)RAND_MAX; }

Population population_create(int size, int num_inputs, int num_outputs) {
    Population p;
    memset(&p, 0, sizeof(p));
    p.size = size;
    p.num_inputs = num_inputs;
    p.num_outputs = num_outputs;
    p.next_node_id = num_inputs + num_outputs;
    p.next_innovation = 0;

    p.genomes = malloc(sizeof(Genome) * (size_t)size);
    for (int i = 0; i < size; i++) {
        p.genomes[i] = genome_create_minimal(num_inputs, num_outputs);
        p.genomes[i].id = i;
    }
    p.best_ever_fitness = -1e18;
    p.has_best_ever = 0;
    return p;
}

void population_free(Population *p) {
    for (int i = 0; i < p->size; i++) genome_free(&p->genomes[i]);
    free(p->genomes);
    for (int s = 0; s < p->num_species; s++) {
        genome_free(&p->species[s].representative);
        free(p->species[s].members);
    }
    free(p->species);
    free(p->innovations);
    if (p->has_best_ever) genome_free(&p->best_ever);
}

void population_evaluate(Population *p, double (*fitness_fn)(const Genome *, void *), void *user_data) {
    for (int i = 0; i < p->size; i++) {
        p->genomes[i].fitness = fitness_fn(&p->genomes[i], user_data);
        if (!p->has_best_ever || p->genomes[i].fitness > p->best_ever_fitness) {
            if (p->has_best_ever) genome_free(&p->best_ever);
            p->best_ever = genome_copy(&p->genomes[i]);
            p->best_ever_fitness = p->genomes[i].fitness;
            p->has_best_ever = 1;
        }
    }
}

Genome *population_best(Population *p) {
    int best = 0;
    for (int i = 1; i < p->size; i++)
        if (p->genomes[i].fitness > p->genomes[best].fitness) best = i;
    return &p->genomes[best];
}

/* sort helper: descending fitness, operating on an array of genome indices */
static Population *g_sort_pop;
static int member_cmp(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    double fa = g_sort_pop->genomes[ia].fitness;
    double fb = g_sort_pop->genomes[ib].fitness;
    if (fa > fb) return -1;
    if (fa < fb) return 1;
    return 0;
}

/* fitness-proportionate-ish pick restricted to the top SURVIVAL_THRESHOLD
   slice of a (fitness-sorted) species member list */
static Genome *select_parent(Population *p, Species *sp, const int *sorted_members) {
    int pool = (int)(sp->num_members * SURVIVAL_THRESHOLD);
    if (pool < 1) pool = 1;
    int pick = sorted_members[rand() % pool];
    return &p->genomes[pick];
}

void population_reproduce(Population *p) {
    adjust_fitness_sharing(p);

    double total_avg = 0;
    for (int s = 0; s < p->num_species; s++) total_avg += p->species[s].avg_fitness;
    if (total_avg <= 0) total_avg = 1;

    Genome *next_gen = malloc(sizeof(Genome) * (size_t)p->size);
    int filled = 0;

    /* zero out the reproductive allotment of species that have stagnated too
       long (but never wipe out every species) */
    if (p->num_species > 1) {
        for (int s = 0; s < p->num_species; s++)
            if (p->species[s].staleness > STALE_SPECIES_LIMIT)
                p->species[s].avg_fitness = 0;
    }

    g_sort_pop = p;
    for (int s = 0; s < p->num_species && filled < p->size; s++) {
        Species *sp = &p->species[s];
        if (sp->num_members == 0) continue;

        int *sorted = malloc(sizeof(int) * (size_t)sp->num_members);
        memcpy(sorted, sp->members, sizeof(int) * (size_t)sp->num_members);
        qsort(sorted, (size_t)sp->num_members, sizeof(int), member_cmp);

        int allotted = (int)((sp->avg_fitness / total_avg) * p->size);
        if (s == 0 && allotted < 1) allotted = 1; /* guarantee at least the top species breeds */

        /* elitism: a healthy species' champion passes through unchanged */
        if (sp->num_members >= 5 && filled < p->size) {
            next_gen[filled++] = genome_copy(&p->genomes[sorted[0]]);
        }

        for (int k = 1; k < allotted && filled < p->size; k++) {
            Genome child;
            if (sp->num_members > 1 && rd() < CROSSOVER_RATE) {
                Genome *pa = select_parent(p, sp, sorted);
                Genome *pb = select_parent(p, sp, sorted);
                int equal = (fabs(pa->fitness - pb->fitness) < 1e-9);
                if (pa->fitness < pb->fitness) { Genome *tmp = pa; pa = pb; pb = tmp; }
                child = crossover(pa, pb, equal);
            } else {
                Genome *pa = select_parent(p, sp, sorted);
                child = genome_copy(pa);
            }
            if (rd() < ADD_CONNECTION_RATE) mutate_add_connection(&child, p);
            if (rd() < ADD_NODE_RATE) mutate_add_node(&child, p);
            mutate_weights(&child);
            child.fitness = 0;
            next_gen[filled++] = child;
        }
        free(sorted);
    }

    /* fill any leftover slots (rounding slack) with mutated copies of
       randomly chosen individuals from the whole population */
    while (filled < p->size) {
        Genome *pa = &p->genomes[rand() % p->size];
        Genome *pb = &p->genomes[rand() % p->size];
        Genome child = genome_copy(pa->fitness >= pb->fitness ? pa : pb);
        mutate_weights(&child);
        next_gen[filled++] = child;
    }

    for (int i = 0; i < p->size; i++) genome_free(&p->genomes[i]);
    free(p->genomes);
    p->genomes = next_gen;
    for (int i = 0; i < p->size; i++) p->genomes[i].id = i;

    p->generation++;
    speciate_population(p);
}
