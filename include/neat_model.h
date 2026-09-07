#ifndef NEAT_MODEL_H
#define NEAT_MODEL_H

#include <stdio.h>
#include <stdlib.h>

typedef enum {
    NEAT_NODE_INPUT,
    NEAT_NODE_OUTPUT,
    NEAT_NODE_HIDDEN,
    NEAT_NODE_BIAS
} NeatNodeType;

typedef struct {
    int id;
    NeatNodeType type;
} NeatNode;

typedef struct {
    int in_node, out_node;
    double weight;
    int enabled;
} NeatConn;

typedef struct {
    int num_inputs;   /* size the caller's `inputs` array to this  */
    int num_outputs;  /* size the caller's `outputs` array to this */
    NeatNode *nodes;
    int num_nodes;
    NeatConn *conns;
    int num_conns;
} NeatModel;

/* Loads a model saved by genome_save() / the neat_xor trainer.
   Returns 1 on success, 0 if the file couldn't be opened or parsed
   (in which case `m` is left zeroed and safe to pass to neat_model_free). */
int neat_model_load(const char *path, NeatModel *m);

/* Frees memory owned by a loaded model. Safe to call on a zeroed model. */
void neat_model_free(NeatModel *m);

/* Runs a forward pass. `inputs` must have m->num_inputs entries (including
   any bias input the model was trained with, typically fixed at 1.0) and
   `outputs` must have room for m->num_outputs entries. */
void neat_model_run(const NeatModel *m, const double *inputs, double *outputs);

#ifdef NEAT_MODEL_IMPLEMENTATION

#include <math.h>
#include <string.h>

static double neat_model_activate(double x) {
    return 1.0 / (1.0 + exp(-4.9 * x)); /* steepened sigmoid, matches training */
}

int neat_model_load(const char *path, NeatModel *m) {
    memset(m, 0, sizeof(*m));
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char tag[32];
    int version;
    if (fscanf(f, "%31s %d", tag, &version) != 2 || strcmp(tag, "NEAT_MODEL") != 0) {
        fclose(f);
        return 0;
    }

    char key[32];
    if (fscanf(f, "%31s %d", key, &m->num_inputs) != 2) { fclose(f); return 0; }
    if (fscanf(f, "%31s %d", key, &m->num_outputs) != 2) { fclose(f); return 0; }

    if (fscanf(f, "%31s %d", key, &m->num_nodes) != 2) { fclose(f); return 0; }
    m->nodes = malloc(sizeof(NeatNode) * (size_t)(m->num_nodes > 0 ? m->num_nodes : 1));
    for (int i = 0; i < m->num_nodes; i++) {
        char typestr[16];
        int id;
        if (fscanf(f, "%31s %d %15s", key, &id, typestr) != 3) { fclose(f); neat_model_free(m); return 0; }
        m->nodes[i].id = id;
        if (!strcmp(typestr, "INPUT")) m->nodes[i].type = NEAT_NODE_INPUT;
        else if (!strcmp(typestr, "OUTPUT")) m->nodes[i].type = NEAT_NODE_OUTPUT;
        else if (!strcmp(typestr, "BIAS")) m->nodes[i].type = NEAT_NODE_BIAS;
        else m->nodes[i].type = NEAT_NODE_HIDDEN;
    }

    if (fscanf(f, "%31s %d", key, &m->num_conns) != 2) { fclose(f); neat_model_free(m); return 0; }
    m->conns = malloc(sizeof(NeatConn) * (size_t)(m->num_conns > 0 ? m->num_conns : 1));
    for (int i = 0; i < m->num_conns; i++) {
        if (fscanf(f, "%31s %d %d %lf %d", key,
                    &m->conns[i].in_node, &m->conns[i].out_node,
                    &m->conns[i].weight, &m->conns[i].enabled) != 5) {
            fclose(f);
            neat_model_free(m);
            return 0;
        }
    }

    fclose(f);
    return 1;
}

void neat_model_free(NeatModel *m) {
    free(m->nodes);
    free(m->conns);
    m->nodes = NULL;
    m->conns = NULL;
    m->num_nodes = m->num_conns = 0;
}

void neat_model_run(const NeatModel *m, const double *inputs, double *outputs) {
    int max_id = 0;
    for (int i = 0; i < m->num_nodes; i++)
        if (m->nodes[i].id > max_id) max_id = m->nodes[i].id;

    int size = max_id + 1;
    double *values = calloc((size_t)size, sizeof(double));
    double *next = calloc((size_t)size, sizeof(double));

    for (int i = 0; i < m->num_inputs; i++) values[i] = inputs[i];

    /* Same relaxation approach as the trainer's network.c: repeatedly
       recompute every non-input node for (num_nodes + 1) passes, which
       fully propagates signal through any feed-forward topology. */
    for (int iter = 0; iter < m->num_nodes + 1; iter++) {
        memcpy(next, values, sizeof(double) * (size_t)size);
        for (int n = 0; n < m->num_nodes; n++) {
            int id = m->nodes[n].id;
            NeatNodeType t = m->nodes[n].type;
            if (t == NEAT_NODE_INPUT || t == NEAT_NODE_BIAS) continue;

            double sum = 0;
            for (int c = 0; c < m->num_conns; c++) {
                if (m->conns[c].enabled && m->conns[c].out_node == id)
                    sum += m->conns[c].weight * values[m->conns[c].in_node];
            }
            next[id] = neat_model_activate(sum);
        }
        memcpy(values, next, sizeof(double) * (size_t)size);
    }

    for (int o = 0; o < m->num_outputs; o++)
        outputs[o] = values[m->num_inputs + o];

    free(values);
    free(next);
}

#endif /* NEAT_MODEL_IMPLEMENTATION */
#endif /* NEAT_MODEL_H */
