#include "neat.h"
#include <string.h>
#include <math.h>

/* steepened sigmoid, as used in the original NEAT paper */
static double activate(double x) {
    return 1.0 / (1.0 + exp(-4.9 * x));
}

/* Relaxation-based feed-forward evaluation: since add-connection mutations
   reject anything that would create a cycle, repeatedly recomputing every
   non-input node's value for (num_nodes + 1) passes is guaranteed to fully
   propagate signal from inputs to outputs, however deep the network is. */
void network_evaluate(const Genome *g, const double *inputs, double *outputs,
                       int num_inputs, int num_outputs) {
    int max_id = 0;
    for (int i = 0; i < g->num_nodes; i++)
        if (g->nodes[i].id > max_id) max_id = g->nodes[i].id;

    int size = max_id + 1;
    double *values = calloc((size_t)size, sizeof(double));
    double *next = calloc((size_t)size, sizeof(double));

    for (int i = 0; i < num_inputs; i++) values[i] = inputs[i];

    for (int iter = 0; iter < g->num_nodes + 1; iter++) {
        memcpy(next, values, sizeof(double) * (size_t)size);
        for (int n = 0; n < g->num_nodes; n++) {
            int id = g->nodes[n].id;
            NodeType t = g->nodes[n].type;
            if (t == NODE_INPUT || t == NODE_BIAS) continue;

            double sum = 0;
            for (int c = 0; c < g->num_conns; c++) {
                if (g->conns[c].enabled && g->conns[c].out_node == id)
                    sum += g->conns[c].weight * values[g->conns[c].in_node];
            }
            next[id] = activate(sum);
        }
        memcpy(values, next, sizeof(double) * (size_t)size);
    }

    for (int o = 0; o < num_outputs; o++)
        outputs[o] = values[num_inputs + o];

    free(values);
    free(next);
}
