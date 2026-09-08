#include <stdio.h>
#include "neat.h"

#define POP_SIZE        150
#define MAX_GENERATIONS 300
#define NUM_INPUTS      3   /* 2 real inputs + 1 bias */
#define NUM_OUTPUTS     1

static double xor_inputs[4][3] = {
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};
static double xor_expected[4] = {0, 1, 1, 0};

Dataset data = {
    .num_samples = 4,
    .num_inputs = NUM_INPUTS,
    .num_outputs = NUM_OUTPUTS,
    .inputs = xor_inputs,
    .expectations = xor_expected
};

int main(int argc, char **argv){
    generate_neural_net(&data);
	return 0;
}