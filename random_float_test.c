#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

float random_float(float Min, float Max)
{
    return (((float)rand() / (float)RAND_MAX) * (Max - Min)) + Min;
}

int main(unsigned int argc, char** argv)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < atoi(argv[1]); i++)
    {
        printf("%f ", random_float(-1.0f, 1.0f));
    }
}