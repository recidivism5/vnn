#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

typedef struct Neuron {
    float* weights;
    unsigned int numWeights;
    float  bias;
    float  dCdZ;
    float  zeta;
    float  activation;
} Neuron;

typedef struct Layer {
    Neuron* neurons;
    unsigned int numNeurons;
} Layer;

typedef struct NeuralNet {
    Layer* layers;
    unsigned int numLayers;
} NeuralNet;

void calculate_activations(Layer* L1, Layer* L2)
{
    unsigned int i;
    unsigned int j;
    float sum;
    for (i = 0; i < L2->numNeurons; i++)
    {
        sum = 0;
        for (j = 0; j < L1->numNeurons; j++)
        {
            sum += L1->neurons[j].activation * L2->neurons[i].weights[j];
        }
        sum += L2->neurons[i].bias;
        L2->neurons[i].zeta = sum;
        L2->neurons[i].activation = fast_sigmoid(sum);
    }
}

void set_input(NeuralNet* netPtr, float* inputBuffer)
{
    unsigned int i;
    //fill input node activations:
    for (i = 0; i < netPtr->layers[0].numNeurons; i++)
    {
        netPtr->layers[0].neurons[i].activation = inputBuffer[i];
    }
}

void propagate_forward(NeuralNet* netPtr)
{
    unsigned int i;
    for (i = 0; i < netPtr->numLayers - 1; i++)
    {
        calculate_activations(&(netPtr->layers[i]), &(netPtr->layers[i+1]));
    }
}

float cost_function(Layer* outputLayer, float* trainingBuffer)
{
    float cost = 0;
    float diff;
    unsigned int i;
    for (i = 0; i < outputLayer->numNeurons; i++)
    {
        diff = outputLayer->neurons[i].activation - trainingBuffer[i];
        cost += diff * diff;
    }
    return cost;
}

void print_output_layer_activations(NeuralNet* netPtr, char expected)
{
    unsigned int i;
    for (i = 0; i < netPtr->layers[netPtr->numLayers-1].numNeurons; i++)
    {
        printf("%f ", netPtr->layers[netPtr->numLayers-1].neurons[i].activation);
    }
    putchar('\n');
    float output[10] = {0.0f};
    output[expected] = 1.0f;
    printf("expected: ");
    for (i = 0; i < 10; i++)
    {
        printf("%f ", output[i]);
    }
    putchar('\n');
    printf("cost = %f\n", cost_function(&(netPtr->layers[netPtr->numLayers-1]), output));
}

NeuralNet* load_network(char* filePath)
{
    FILE* fptr = fopen(filePath, "rb");
    NeuralNet* netPtr = malloc(sizeof(NeuralNet));
    fread(&(netPtr->numLayers), sizeof(unsigned int), 1, fptr);
    printf("numLayers = %u\n", netPtr->numLayers);
    netPtr->layers = malloc(netPtr->numLayers * sizeof(Layer));
    unsigned int i;
    unsigned int j;
    unsigned int k;
    for (i = 0; i < netPtr->numLayers; i++)
    {
        fread(&(netPtr->layers[i].numNeurons), sizeof(unsigned int), 1, fptr);
        printf("layer %u numNeurons = %u\n", i, netPtr->layers[i].numNeurons);
        netPtr->layers[i].neurons = malloc(netPtr->layers[i].numNeurons * sizeof(Neuron));
    }
    for (i = 1; i < netPtr->numLayers; i++)
    {
        for (j = 0; j < netPtr->layers[i].numNeurons; j++)
        {
            fread(&(netPtr->layers[i].neurons[j].bias), sizeof(float), 1, fptr);
            //printf("layer %u neuron %u bias = %f\n", i, j, netPtr->layers[i].neurons[j].bias); 
            netPtr->layers[i].neurons[j].numWeights = netPtr->layers[i-1].numNeurons;
            netPtr->layers[i].neurons[j].weights = malloc(netPtr->layers[i].neurons[j].numWeights * sizeof(float));
            fread(netPtr->layers[i].neurons[j].weights, netPtr->layers[i].neurons[j].numWeights * sizeof(float), 1, fptr);
        }
    }
    fclose(fptr);
    return netPtr;
}

void run_test(NeuralNet* netPtr, char* inputFilePath, char* labelFilePath, unsigned int numCases)
{
    FILE* fptrI = fopen(inputFilePath, "rb");
    fseek(fptrI, 16, SEEK_SET);
    FILE* fptrL = fopen(labelFilePath, "rb");
    fseek(fptrL, 8, SEEK_SET);

    float* inputBuffer = malloc(28*28*sizeof(float));
    unsigned int i;
    unsigned int j;
    for (i = 0; i < numCases; i++)
    {
        for (j = 0; j < (28*28); j++)
        {
            inputBuffer[j] = (float)fgetc(fptrI) / (float)255.0f;
        }
        set_input(netPtr, inputBuffer);
        propagate_forward(netPtr);
        print_output_layer_activations(netPtr, fgetc(fptrL));
        printf("\n");
    }
    fclose(fptrI);
    fclose(fptrL);
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        printf("Usage: ./net_runner <*.network> <*.idx3-ubyte> <*.idx1-ubyte> <numCases>\n");
        return 1;
    }
    printf("Loading %s\n", argv[1]);
    NeuralNet* netPtr = load_network(argv[1]);
    printf("Done.\nRunning tests...\n");
    unsigned int numImages;
    sscanf(argv[4], "%u", &numImages);
    run_test(netPtr, argv[2], argv[3], numImages);
    printf("Done.\n");
}