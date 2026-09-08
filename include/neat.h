#ifndef NEAT_H
#define NEAT_H

#include <stdlib.h>

/* ---------------- Configuration / hyperparameters ---------------- */
#define WEIGHT_MUTATE_RATE      0.8   /* chance a genome's weights get touched at all */
#define WEIGHT_PERTURB_RATE     0.9   /* within that, chance of nudge vs full reset  */
#define WEIGHT_PERTURB_STEP     0.5
#define ADD_CONNECTION_RATE     0.08
#define ADD_NODE_RATE           0.03
#define DISABLE_GENE_RATE       0.75  /* prob. a gene disabled in either parent stays disabled */
#define CROSSOVER_RATE          0.75

#define COMPAT_C1  1.0   /* excess genes coefficient      */
#define COMPAT_C2  1.0   /* disjoint genes coefficient     */
#define COMPAT_C3  0.4   /* weight difference coefficient  */
#define COMPAT_THRESHOLD 3.0

#define STALE_SPECIES_LIMIT 15
#define SURVIVAL_THRESHOLD  0.2   /* fraction of each species allowed to reproduce */

/* ---------------- Core genome representation ---------------- */

typedef enum { NODE_INPUT, NODE_OUTPUT, NODE_HIDDEN, NODE_BIAS } NodeType;

typedef struct {
    int num_samples;
    int num_inputs;
    int num_outputs;
    double (*inputs)[2];
    double *expectations;
} Dataset;

typedef struct {
    int id;
    NodeType type;
} NodeGene;

typedef struct {
    int innovation;
    int in_node;
    int out_node;
    double weight;
    int enabled;
} ConnectionGene;

typedef struct {
    int id;
    NodeGene *nodes;
    int num_nodes, node_cap;
    ConnectionGene *conns;
    int num_conns, conn_cap;
    double fitness;
    double adjusted_fitness;
    int species_id;
} Genome;

typedef struct {
    int id;
    Genome representative;
    int *members;          /* indices into Population.genomes */
    int num_members, member_cap;
    double best_fitness;
    double avg_fitness;
    int staleness;          /* generations since last improvement */
} Species;

typedef struct {
    int in_node, out_node, innovation;
} InnovationRecord;

typedef struct {
    Genome *genomes;
    int size;

    Species *species;
    int num_species, species_cap;

    int generation;
    int next_node_id;
    int next_innovation;

    InnovationRecord *innovations;
    int num_innovations, innov_cap;

    int num_inputs;   /* includes bias input */
    int num_outputs;

    Genome best_ever;
    double best_ever_fitness;
    int has_best_ever;
} Population;

/* ---------------- genome.c ---------------- */
Genome genome_create_minimal(int num_inputs, int num_outputs);
Genome genome_copy(const Genome *g);
void   genome_free(Genome *g);
void   genome_add_node(Genome *g, int id, NodeType type);
void   genome_add_conn(Genome *g, int innovation, int in, int out, double weight, int enabled);
int    genome_has_conn(const Genome *g, int in, int out);
int    genome_path_exists(const Genome *g, int from, int to);

void   mutate_weights(Genome *g);
void   mutate_add_connection(Genome *g, Population *p);
void   mutate_add_node(Genome *g, Population *p);
Genome crossover(const Genome *fitter, const Genome *other, int equal_fitness);
double genome_distance(const Genome *a, const Genome *b);

int get_innovation(Population *p, int in_node, int out_node);

/* ---------------- network.c ---------------- */
void network_evaluate(const Genome *g, const double *inputs, double *outputs,
                       int num_inputs, int num_outputs);

/* ---------------- species.c ---------------- */
void speciate_population(Population *p);
void adjust_fitness_sharing(Population *p);

/* ---------------- population.c ---------------- */
Population population_create(int size, int num_inputs, int num_outputs);
void population_free(Population *p);
void population_evaluate(Population *p, double (*fitness_fn)(const Genome *, void *), void *user_data);
void population_reproduce(Population *p);
Genome *population_best(Population *p);

/* ---------------- genome_io.c ---------------- */
/* Save/load a trained genome to a portable text format ("*.neat" file).
   The same format is read by the standalone include/neat_model.h header,
   so a model saved here can be loaded from any other C program without
   pulling in the rest of this project. */
int genome_save(const Genome *g, int num_inputs, int num_outputs, const char *path);
int genome_load(Genome *g, int *num_inputs, int *num_outputs, const char *path);

/* ---------------- generate.c ---------------- */
void generate_neural_net(void *user_data);

#endif /* NEAT_H */
