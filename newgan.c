#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float fast_sigmoid_derivative(float fx)
{
    return fx * (1-fx);
}

float random_float(float Min, float Max)
{
    return (((float)rand() / (float)RAND_MAX) * (Max - Min)) + Min;
}

unsigned int random_uint(unsigned int max)
{
    return (unsigned int)(rand() % max);
}

typedef struct Layer {
    unsigned int numNodes;
    float* activations;
    float* biases;
    float* weightMatrix;
    unsigned int weightMatrixSize;
    float* dCdZs;
} Layer;

typedef struct NeuralNet {
    Layer* layers;
    unsigned int numLayers;
} NeuralNet;

typedef struct TrainingData {
    float* inputData;
    float* outputData;
    unsigned int inputSize;
    unsigned int outputSize;
    unsigned int numPairs;
} TrainingData;

NeuralNet* gen_neural_net(unsigned int numLayers, unsigned int* layerHeights)
{
    NeuralNet* netPtr = malloc(sizeof(NeuralNet));
    netPtr->numLayers = numLayers;
    netPtr->layers = malloc(numLayers * sizeof(Layer));
    unsigned int i;
    netPtr->layers[0].activations = malloc(layerHeights[0] * sizeof(float));
    netPtr->layers[0].numNodes = layerHeights[0];
    for (i = 1; i < numLayers; i++)
    {
        netPtr->layers[i].numNodes = layerHeights[i];
        netPtr->layers[i].weightMatrixSize = layerHeights[i] * layerHeights[i-1];

        netPtr->layers[i].activations = malloc(layerHeights[i] * sizeof(float));
        netPtr->layers[i].biases = malloc(layerHeights[i] * sizeof(float));
        netPtr->layers[i].dCdZs = malloc(layerHeights[i] * sizeof(float));
        netPtr->layers[i].weightMatrix = malloc(layerHeights[i] * layerHeights[i-1] * sizeof(float));
    }
    return netPtr;
}

void randomize_network(NeuralNet* netPtr)
{
    unsigned int i;
    unsigned int j;
    unsigned int k;
    
    for (i = 1; i < netPtr->numLayers; i++)
    {
        for (j = 0; j < netPtr->layers[i].numNodes; j++)
        {
            netPtr->layers[i].biases[j] = random_float(-1.0f, 1.0f);
            for (k = 0; k < netPtr->layers[i].weightMatrixSize; k++)
            {
                netPtr->layers[i].weightMatrix[k] = random_float(-1.0f, 1.0f);
            }
        }
    }
}

void propagate_forward(NeuralNet* netPtr)
{
    
}

void save_network_to_disk(NeuralNet* netPtr, char* filePath)
{
    FILE* fptr = fopen(filePath, "wb");
    fwrite(&(netPtr->numLayers), sizeof(netPtr->numLayers), 1, fptr);
    
    unsigned int i;
    unsigned int j;
    unsigned int k;
    
    for (i = 0; i < netPtr->numLayers; i++)
    {
        fwrite(&(netPtr->layers[i].numNodes), sizeof(netPtr->layers[i].numNodes), 1, fptr);
    }
    for (i = 1; i < netPtr->numLayers; i++)
    {
        fwrite(netPtr->layers[i].biases, netPtr->layers[i].numNodes * sizeof(float), 1, fptr);
        fwrite(netPtr->layers[i].weightMatrix, netPtr->layers[i].weightMatrixSize * sizeof(float), 1, fptr);
    }
    fclose(fptr);
}

NeuralNet* load_network_from_disk(char* filePath)
{
    NeuralNet* netPtr = malloc(sizeof(NeuralNet));
    FILE* fptr = fopen(filePath, "rb");
    fread(&(netPtr->numLayers), sizeof(unsigned int), 1, fptr);
    printf("numLayers = %u\n", netPtr->numLayers);
    unsigned int i;

    netPtr->layers = malloc(netPtr->numLayers * sizeof(Layer));
    for (i = 0; i < netPtr->numLayers; i++)
    {
        fread(&(netPtr->layers[i].numNodes), sizeof(unsigned int), 1, fptr);
        printf("Layer %u nodes: %u\n", i, netPtr->layers[i].numNodes);
    }
    netPtr->layers[0].activations = malloc(netPtr->layers[0].numNodes * sizeof(float));
    for (i = 1; i < netPtr->numLayers; i++)
    {
        netPtr->layers[i].weightMatrixSize = netPtr->layers[i].numNodes * netPtr->layers[i-1].numNodes;
        netPtr->layers[i].activations = malloc(netPtr->layers[i].numNodes * sizeof(float));
        netPtr->layers[i].biases = malloc(netPtr->layers[i].numNodes * sizeof(float));
        netPtr->layers[i].dCdZs = malloc(netPtr->layers[i].numNodes * sizeof(float));
        netPtr->layers[i].weightMatrix = malloc(netPtr->layers[i].weightMatrixSize * sizeof(float));

        fread(netPtr->layers[i].biases, netPtr->layers[i].numNodes * sizeof(float), 1, fptr);
        fread(netPtr->layers[i].weightMatrix, netPtr->layers[i].weightMatrixSize * sizeof(float), 1, fptr);
    }
    fclose(fptr);
    return netPtr;
}

void print_layer_activations(NeuralNet* netPtr, unsigned int layerNumber)
{
    unsigned int i;
    for (i = 0; i < netPtr->layers[layerNumber].numNodes; i++)
    {
        printf("%f ", netPtr->layers[layerNumber].activations[i]);
    }
    putchar('\n');
}

void print_expected_output_buffer(TrainingData* tDataPtr, unsigned int startIndex)
{
    unsigned int j;
    unsigned int outputIndex;
    outputIndex = startIndex * tDataPtr->outputSize;
    for (j = 0; j < tDataPtr->outputSize; j++)
    {
        printf("%f ", tDataPtr->outputData[outputIndex + j]);
    }
    putchar('\n');
}

void test_network(NeuralNet* netPtr, TrainingData* tDataPtr, unsigned int startIndex, unsigned int numCases)
{
    unsigned int i;
    for (i = 0; i < numCases; i++)
    {
        memcpy(netPtr->layers[0].activations, tDataPtr->inputData + ((startIndex + i) * tDataPtr->inputSize), tDataPtr->inputSize * sizeof(float));
        propagate_forward(netPtr);
        printf("Output: ");
        print_layer_activations(netPtr, netPtr->numLayers-1);
        printf("Expect: ");
        print_expected_output_buffer(tDataPtr, startIndex+i);
    }
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    printf("Loading training data...\n");
    TrainingData* tDataPtr = load_training_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 60000, 28*28, 10, 16, 8, "char", "char", "highest-node");
    printf("Done.\n");

    NeuralNet* netPtr;
    if (argc > 1)
    {
        printf("Loading %s\n", argv[1]);
        netPtr = load_network_from_disk(argv[1]);
        printf("Done.\n");
    }
    else
    {
        printf("Generating new neural network with default settings...\n");
        unsigned int layerHeights[] = {28*28, 16, 16, 10};
        netPtr = gen_neural_net(4, layerHeights);
        printf("Randomizing weights and biases...\n");
        randomize_network(netPtr);
        printf("Done.\n");
        printf("Save .network file as? ");
        char outName[30];
        scanf("%s", outName);
        strcat(outName, ".network");
        save_network_to_disk(netPtr, outName);
        return 0;
    }

    if (argc < 5)
    {
        printf("Useage: ./vectorized_nn <*.network> <-train / -test> <numEpochs / trainingStartIndex> <learningRate / numTrainingPairs>\n");
        return 0;
    }

    if (!strcmp(argv[2], "-train"))
    {
        unsigned int epochCount;
        float trainingRate;
        sscanf(argv[3], "%u", &epochCount);
        sscanf(argv[4], "%f", &trainingRate);
        printf("Training network for %u epochs with trainingRate = %f\n", epochCount, trainingRate);
        train(netPtr, tDataPtr, 10000, epochCount, trainingRate);
        printf("Training complete.\n");
        printf("Save .network file as? ");
        char outName[30];
        scanf("%s", outName);
        strcat(outName, ".network");
        save_network_to_disk(netPtr, outName);
        return 0;
    }
    else if (!strcmp(argv[2], "-test"))
    {
        unsigned int startIndex;
        unsigned int pairCount;
        sscanf(argv[3], "%u", &startIndex);
        sscanf(argv[4], "%u", &pairCount);
        printf("Testing network on pair %u through %u\n", startIndex, startIndex + pairCount);
        free(tDataPtr);
        tDataPtr = load_training_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000, 28*28, 10, 16, 8, "char", "char", "highest-node");
        test_network(netPtr, tDataPtr, startIndex, pairCount);
        return 0;
    }
}