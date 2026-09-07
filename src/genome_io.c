#include "neat.h"
#include <stdio.h>
#include <string.h>

/* File format (plain text, one field per line, whitespace-separated):

   NEAT_MODEL 1
   inputs <N>
   outputs <N>
   num_nodes <N>
   node <id> <INPUT|OUTPUT|HIDDEN|BIAS>
   ... one line per node ...
   num_conns <N>
   conn <in_node> <out_node> <weight> <enabled 0|1>
   ... one line per connection ...

   This is intentionally simple and human-readable rather than binary, so
   it's easy to inspect, diff, or hand-edit, and trivial to parse from any
   language. include/neat_model.h reads this exact format independently. */

static const char *node_type_str(NodeType t) {
    switch (t) {
        case NODE_INPUT:  return "INPUT";
        case NODE_OUTPUT: return "OUTPUT";
        case NODE_BIAS:   return "BIAS";
        default:          return "HIDDEN";
    }
}

int genome_save(const Genome *g, int num_inputs, int num_outputs, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;

    fprintf(f, "NEAT_MODEL 1\n");
    fprintf(f, "inputs %d\n", num_inputs);
    fprintf(f, "outputs %d\n", num_outputs);

    fprintf(f, "num_nodes %d\n", g->num_nodes);
    for (int i = 0; i < g->num_nodes; i++)
        fprintf(f, "node %d %s\n", g->nodes[i].id, node_type_str(g->nodes[i].type));

    fprintf(f, "num_conns %d\n", g->num_conns);
    for (int i = 0; i < g->num_conns; i++)
        fprintf(f, "conn %d %d %.10f %d\n",
                g->conns[i].in_node, g->conns[i].out_node,
                g->conns[i].weight, g->conns[i].enabled);

    fclose(f);
    return 1;
}

int genome_load(Genome *g, int *num_inputs, int *num_outputs, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    memset(g, 0, sizeof(*g));
    char tag[32];
    int version;
    if (fscanf(f, "%31s %d", tag, &version) != 2 || strcmp(tag, "NEAT_MODEL") != 0) {
        fclose(f);
        return 0;
    }

    char key[32];
    if (fscanf(f, "%31s %d", key, num_inputs) != 2) { fclose(f); return 0; }
    if (fscanf(f, "%31s %d", key, num_outputs) != 2) { fclose(f); return 0; }

    int num_nodes = 0;
    if (fscanf(f, "%31s %d", key, &num_nodes) != 2) { fclose(f); return 0; }
    for (int i = 0; i < num_nodes; i++) {
        int id;
        char typestr[16];
        if (fscanf(f, "%31s %d %15s", key, &id, typestr) != 3) { fclose(f); genome_free(g); return 0; }
        NodeType t = NODE_HIDDEN;
        if (!strcmp(typestr, "INPUT")) t = NODE_INPUT;
        else if (!strcmp(typestr, "OUTPUT")) t = NODE_OUTPUT;
        else if (!strcmp(typestr, "BIAS")) t = NODE_BIAS;
        genome_add_node(g, id, t);
    }

    int num_conns = 0;
    if (fscanf(f, "%31s %d", key, &num_conns) != 2) { fclose(f); genome_free(g); return 0; }
    for (int i = 0; i < num_conns; i++) {
        int in, out, enabled;
        double weight;
        if (fscanf(f, "%31s %d %d %lf %d", key, &in, &out, &weight, &enabled) != 5) {
            fclose(f);
            genome_free(g);
            return 0;
        }
        genome_add_conn(g, i, in, out, weight, enabled); /* innovation # doesn't matter post-training */
    }

    fclose(f);
    return 1;
}
