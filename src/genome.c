#include "neat.h"
#include <string.h>
#include <math.h>

#define MAX_NODE_ID_BOUND 8192  /* generous fixed bound for reachability scratch space */

static double rand_uniform(void) { return (double)rand() / (double)RAND_MAX; }
static double rand_range(double lo, double hi) { return lo + rand_uniform() * (hi - lo); }
static double rand_weight(void) { return rand_range(-1.0, 1.0); }

static void ensure_node_cap(Genome *g) {
    if (g->num_nodes >= g->node_cap) {
        g->node_cap = g->node_cap ? g->node_cap * 2 : 8;
        g->nodes = realloc(g->nodes, sizeof(NodeGene) * g->node_cap);
    }
}
static void ensure_conn_cap(Genome *g) {
    if (g->num_conns >= g->conn_cap) {
        g->conn_cap = g->conn_cap ? g->conn_cap * 2 : 8;
        g->conns = realloc(g->conns, sizeof(ConnectionGene) * g->conn_cap);
    }
}

void genome_add_node(Genome *g, int id, NodeType type) {
    for (int i = 0; i < g->num_nodes; i++)
        if (g->nodes[i].id == id) return; /* already present */
    ensure_node_cap(g);
    g->nodes[g->num_nodes].id = id;
    g->nodes[g->num_nodes].type = type;
    g->num_nodes++;
}

void genome_add_conn(Genome *g, int innovation, int in, int out, double weight, int enabled) {
    ensure_conn_cap(g);
    g->conns[g->num_conns].innovation = innovation;
    g->conns[g->num_conns].in_node = in;
    g->conns[g->num_conns].out_node = out;
    g->conns[g->num_conns].weight = weight;
    g->conns[g->num_conns].enabled = enabled;
    g->num_conns++;
}

int genome_has_conn(const Genome *g, int in, int out) {
    for (int i = 0; i < g->num_conns; i++)
        if (g->conns[i].in_node == in && g->conns[i].out_node == out) return 1;
    return 0;
}

/* Is `to` reachable from `from` following currently-enabled connections?
   Used to reject mutations that would introduce a cycle (we keep XOR
   networks strictly feed-forward). */
int genome_path_exists(const Genome *g, int from, int to) {
    if (from == to) return 1;
    if (from < 0 || from >= MAX_NODE_ID_BOUND || to < 0 || to >= MAX_NODE_ID_BOUND) return 0;

    static char visited[MAX_NODE_ID_BOUND];
    static int stack[MAX_NODE_ID_BOUND];
    memset(visited, 0, sizeof(visited));
    int sp = 0;
    stack[sp++] = from;

    while (sp > 0) {
        int cur = stack[--sp];
        if (cur == to) return 1;
        if (cur < 0 || cur >= MAX_NODE_ID_BOUND || visited[cur]) continue;
        visited[cur] = 1;
        for (int i = 0; i < g->num_conns; i++) {
            if (g->conns[i].enabled && g->conns[i].in_node == cur) {
                int nxt = g->conns[i].out_node;
                if (nxt >= 0 && nxt < MAX_NODE_ID_BOUND && !visited[nxt] && sp < MAX_NODE_ID_BOUND)
                    stack[sp++] = nxt;
            }
        }
    }
    return 0;
}

Genome genome_create_minimal(int num_inputs, int num_outputs) {
    Genome g;
    memset(&g, 0, sizeof(g));

    /* node ids 0..num_inputs-1 = inputs (caller includes a bias input),
       node ids num_inputs..num_inputs+num_outputs-1 = outputs */
    for (int i = 0; i < num_inputs; i++)
        genome_add_node(&g, i, NODE_INPUT);
    for (int o = 0; o < num_outputs; o++)
        genome_add_node(&g, num_inputs + o, NODE_OUTPUT);

    int innov = 0;
    for (int i = 0; i < num_inputs; i++)
        for (int o = 0; o < num_outputs; o++)
            genome_add_conn(&g, innov++, i, num_inputs + o, rand_weight(), 1);

    g.fitness = 0;
    g.species_id = -1;
    return g;
}

Genome genome_copy(const Genome *src) {
    Genome g;
    memset(&g, 0, sizeof(g));
    g.node_cap = src->num_nodes;
    g.num_nodes = src->num_nodes;
    g.nodes = malloc(sizeof(NodeGene) * (g.node_cap ? g.node_cap : 1));
    memcpy(g.nodes, src->nodes, sizeof(NodeGene) * src->num_nodes);

    g.conn_cap = src->num_conns;
    g.num_conns = src->num_conns;
    g.conns = malloc(sizeof(ConnectionGene) * (g.conn_cap ? g.conn_cap : 1));
    memcpy(g.conns, src->conns, sizeof(ConnectionGene) * src->num_conns);

    g.fitness = src->fitness;
    g.adjusted_fitness = src->adjusted_fitness;
    g.species_id = src->species_id;
    g.id = src->id;
    return g;
}

void genome_free(Genome *g) {
    free(g->nodes);
    free(g->conns);
    g->nodes = NULL;
    g->conns = NULL;
    g->num_nodes = g->num_conns = g->node_cap = g->conn_cap = 0;
}

int get_innovation(Population *p, int in_node, int out_node) {
    for (int i = 0; i < p->num_innovations; i++)
        if (p->innovations[i].in_node == in_node && p->innovations[i].out_node == out_node)
            return p->innovations[i].innovation;

    if (p->num_innovations >= p->innov_cap) {
        p->innov_cap = p->innov_cap ? p->innov_cap * 2 : 32;
        p->innovations = realloc(p->innovations, sizeof(InnovationRecord) * p->innov_cap);
    }
    int innov = p->next_innovation++;
    p->innovations[p->num_innovations].in_node = in_node;
    p->innovations[p->num_innovations].out_node = out_node;
    p->innovations[p->num_innovations].innovation = innov;
    p->num_innovations++;
    return innov;
}

void mutate_weights(Genome *g) {
    for (int i = 0; i < g->num_conns; i++) {
        if (rand_uniform() < WEIGHT_MUTATE_RATE) {
            if (rand_uniform() < WEIGHT_PERTURB_RATE) {
                g->conns[i].weight += rand_range(-WEIGHT_PERTURB_STEP, WEIGHT_PERTURB_STEP);
                if (g->conns[i].weight > 8) g->conns[i].weight = 8;
                if (g->conns[i].weight < -8) g->conns[i].weight = -8;
            } else {
                g->conns[i].weight = rand_weight() * 2.0;
            }
        }
    }
}

void mutate_add_connection(Genome *g, Population *p) {
    for (int attempt = 0; attempt < 20; attempt++) {
        int a = g->nodes[rand() % g->num_nodes].id;
        int b = g->nodes[rand() % g->num_nodes].id;

        NodeType ta = NODE_HIDDEN, tb = NODE_HIDDEN;
        for (int i = 0; i < g->num_nodes; i++) {
            if (g->nodes[i].id == a) ta = g->nodes[i].type;
            if (g->nodes[i].id == b) tb = g->nodes[i].type;
        }
        if (tb == NODE_INPUT || tb == NODE_BIAS) continue; /* can't feed into an input */
        if (ta == NODE_OUTPUT) continue;                   /* keep outputs as pure sinks */
        if (a == b) continue;
        if (genome_has_conn(g, a, b)) continue;
        if (genome_path_exists(g, b, a)) continue;          /* would create a cycle */

        int innov = get_innovation(p, a, b);
        genome_add_conn(g, innov, a, b, rand_weight(), 1);
        return;
    }
}

void mutate_add_node(Genome *g, Population *p) {
    if (g->num_conns == 0) return;

    int idx = -1;
    for (int tries = 0; tries < 20 && idx == -1; tries++) {
        int c = rand() % g->num_conns;
        if (g->conns[c].enabled) idx = c;
    }
    if (idx == -1) return;

    g->conns[idx].enabled = 0;
    int in = g->conns[idx].in_node, out = g->conns[idx].out_node;
    double old_weight = g->conns[idx].weight;

    int new_id = p->next_node_id++;
    genome_add_node(g, new_id, NODE_HIDDEN);

    int innov1 = get_innovation(p, in, new_id);
    int innov2 = get_innovation(p, new_id, out);
    genome_add_conn(g, innov1, in, new_id, 1.0, 1);        /* in->new: weight 1 preserves signal */
    genome_add_conn(g, innov2, new_id, out, old_weight, 1); /* new->out: keeps old effect roughly */
}

static const ConnectionGene *find_conn(const Genome *g, int innovation) {
    for (int i = 0; i < g->num_conns; i++)
        if (g->conns[i].innovation == innovation) return &g->conns[i];
    return NULL;
}

Genome crossover(const Genome *fitter, const Genome *other, int equal_fitness) {
    Genome child;
    memset(&child, 0, sizeof(child));

    for (int i = 0; i < fitter->num_nodes; i++)
        genome_add_node(&child, fitter->nodes[i].id, fitter->nodes[i].type);
    if (equal_fitness) {
        for (int i = 0; i < other->num_nodes; i++)
            genome_add_node(&child, other->nodes[i].id, other->nodes[i].type);
    }

    for (int i = 0; i < fitter->num_conns; i++) {
        const ConnectionGene *cf = &fitter->conns[i];
        const ConnectionGene *co = find_conn(other, cf->innovation);
        ConnectionGene chosen = *cf;
        if (co) {
            chosen = (rand_uniform() < 0.5) ? *cf : *co;
            chosen.enabled = ((!cf->enabled || !co->enabled) && rand_uniform() < DISABLE_GENE_RATE) ? 0 : 1;
        }
        /* disjoint/excess genes always inherited from the fitter parent */
        genome_add_conn(&child, chosen.innovation, chosen.in_node, chosen.out_node,
                         chosen.weight, chosen.enabled);
    }

    if (equal_fitness) {
        for (int i = 0; i < other->num_conns; i++) {
            if (!find_conn(fitter, other->conns[i].innovation) &&
                !find_conn(&child, other->conns[i].innovation)) {
                genome_add_conn(&child, other->conns[i].innovation, other->conns[i].in_node,
                                 other->conns[i].out_node, other->conns[i].weight,
                                 other->conns[i].enabled);
            }
        }
    }

    child.fitness = 0;
    child.species_id = -1;
    return child;
}

double genome_distance(const Genome *a, const Genome *b) {
    int matching = 0, disjoint = 0, excess = 0;
    double weight_diff_sum = 0;

    int max_innov_a = 0, max_innov_b = 0;
    for (int i = 0; i < a->num_conns; i++) if (a->conns[i].innovation > max_innov_a) max_innov_a = a->conns[i].innovation;
    for (int i = 0; i < b->num_conns; i++) if (b->conns[i].innovation > max_innov_b) max_innov_b = b->conns[i].innovation;

    size_t matched_b_len = (b->num_conns > 0) ? (size_t)b->num_conns : (size_t)1;
    int *matched_b = calloc(matched_b_len, sizeof(int));
    for (int i = 0; i < a->num_conns; i++) {
        int found = -1;
        for (int j = 0; j < b->num_conns; j++) {
            if (b->conns[j].innovation == a->conns[i].innovation) { found = j; break; }
        }
        if (found >= 0) {
            matching++;
            matched_b[found] = 1;
            weight_diff_sum += fabs(a->conns[i].weight - b->conns[found].weight);
        } else if (a->conns[i].innovation > max_innov_b) {
            excess++;
        } else {
            disjoint++;
        }
    }
    for (int j = 0; j < b->num_conns; j++) {
        if (!matched_b[j]) {
            if (b->conns[j].innovation > max_innov_a) excess++;
            else disjoint++;
        }
    }
    free(matched_b);

    int n = a->num_conns > b->num_conns ? a->num_conns : b->num_conns;
    if (n < 20) n = 1; /* classic NEAT convention: don't normalize small genomes */

    double avg_weight_diff = matching ? weight_diff_sum / matching : 0;
    return (COMPAT_C1 * excess) / n + (COMPAT_C2 * disjoint) / n + COMPAT_C3 * avg_weight_diff;
}
