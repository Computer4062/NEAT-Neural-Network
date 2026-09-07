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

int main(int argc, char **argv){
	generate_neural_net(POP_SIZE, NUM_INPUTS, 4, NUM_OUTPUTS, MAX_GENERATIONS, xor_inputs, xor_expected);
	return 0;
}