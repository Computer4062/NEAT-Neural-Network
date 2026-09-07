#include "neat.h"
#include <string.h>

static void species_add_member(Species *s, int idx) {
    if (s->num_members >= s->member_cap) {
        s->member_cap = s->member_cap ? s->member_cap * 2 : 8;
        s->members = realloc(s->members, sizeof(int) * s->member_cap);
    }
    s->members[s->num_members++] = idx;
}

void speciate_population(Population *p) {
    for (int i = 0; i < p->num_species; i++) p->species[i].num_members = 0;

    for (int i = 0; i < p->size; i++) {
        Genome *g = &p->genomes[i];
        int placed = 0;
        for (int s = 0; s < p->num_species; s++) {
            double d = genome_distance(g, &p->species[s].representative);
            if (d < COMPAT_THRESHOLD) {
                species_add_member(&p->species[s], i);
                g->species_id = p->species[s].id;
                placed = 1;
                break;
            }
        }
        if (!placed) {
            if (p->num_species >= p->species_cap) {
                p->species_cap = p->species_cap ? p->species_cap * 2 : 8;
                p->species = realloc(p->species, sizeof(Species) * p->species_cap);
            }
            Species *s = &p->species[p->num_species];
            memset(s, 0, sizeof(Species));
            s->id = p->num_species;
            s->representative = genome_copy(g);
            species_add_member(s, i);
            g->species_id = s->id;
            p->num_species++;
        }
    }

    /* drop species that lost all members, refresh representative of survivors */
    int write = 0;
    for (int s = 0; s < p->num_species; s++) {
        if (p->species[s].num_members > 0) {
            int rep_idx = p->species[s].members[rand() % p->species[s].num_members];
            genome_free(&p->species[s].representative);
            p->species[s].representative = genome_copy(&p->genomes[rep_idx]);
            p->species[write] = p->species[s];
            write++;
        } else {
            genome_free(&p->species[s].representative);
            free(p->species[s].members);
        }
    }
    p->num_species = write;
}

void adjust_fitness_sharing(Population *p) {
    for (int s = 0; s < p->num_species; s++) {
        Species *sp = &p->species[s];
        double sum = 0, best = -1e18;
        for (int m = 0; m < sp->num_members; m++) {
            Genome *g = &p->genomes[sp->members[m]];
            g->adjusted_fitness = g->fitness / sp->num_members;
            sum += g->fitness;
            if (g->fitness > best) best = g->fitness;
        }
        sp->avg_fitness = sp->num_members ? sum / sp->num_members : 0;
        if (best > sp->best_fitness) {
            sp->best_fitness = best;
            sp->staleness = 0;
        } else {
            sp->staleness++;
        }
    }
}
