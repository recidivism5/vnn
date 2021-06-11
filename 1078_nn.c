#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

void print_network_dimensions(unsigned int numLayers, unsigned int* layerHeights)
{
    printf("DIMENSIONS:\n");
    for (unsigned int i = 0; i < numLayers; i++)
    {
        printf("Layer %i numNodes = %i\n", i, layerHeights[i]);
    }
}

NeuralNet* gen_neural_net(unsigned int numLayers, unsigned int* layerHeights)
{
    print_network_dimensions(numLayers, layerHeights);

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
        }
        for (k = 0; k < netPtr->layers[i].weightMatrixSize; k++)
        {
            netPtr->layers[i].weightMatrix[k] = random_float(-1.0f, 1.0f);
        }
    }
}

void propagate_forward(NeuralNet* netPtr)
{
    unsigned int i;
    unsigned int j;
    unsigned int k;

    for (i = 1; i < netPtr->numLayers; i++)
    {
        for (j = 0; j < netPtr->layers[i].numNodes; j++)
        {
            netPtr->layers[i].activations[j] = 0.0f;
            for (k = 0; k < netPtr->layers[i-1].numNodes; k++)
            {
                netPtr->layers[i].activations[j] += netPtr->layers[i].weightMatrix[j * netPtr->layers[i-1].numNodes + k] * netPtr->layers[i-1].activations[k];
            }
            netPtr->layers[i].activations[j] += netPtr->layers[i].biases[j];
            netPtr->layers[i].activations[j] = fast_sigmoid(netPtr->layers[i].activations[j]);
        }
    }
}

void propagate_backward(NeuralNet* netPtr, float* expectedOutputBuffer, float learningRate)
{
    unsigned int i;
    unsigned int j;
    unsigned int k;

    float dCda;

    unsigned int last;
    last = netPtr->numLayers - 1;
    for (i = 0; i < netPtr->layers[last].numNodes; i++)     //get dCdZs for output layer
    {
        netPtr->layers[last].dCdZs[i] = 2 * (netPtr->layers[last].activations[i] - expectedOutputBuffer[i]) * fast_sigmoid_derivative(netPtr->layers[last].activations[i]) / netPtr->layers[last].numNodes;
    }

    for (i = last - 1; i > 0; i--)      //now get dCdZs for hidden layers
    {
        for (j = 0; j < netPtr->layers[i].numNodes; j++)
        {
            dCda = 0.0f;
            for (k = 0; k < netPtr->layers[i+1].numNodes; k++)
            {
                dCda += netPtr->layers[i+1].dCdZs[k] * netPtr->layers[i+1].weightMatrix[k * netPtr->layers[i].numNodes + j];
            }
            netPtr->layers[i].dCdZs[j] = dCda * fast_sigmoid_derivative(netPtr->layers[i].activations[j]);
        }
    }

    for (i = last; i > 0; i--)      //now calculate and apply changes to each weight and bias
    {
        for (j = 0; j < netPtr->layers[i].numNodes; j++)
        {
            for (k = 0; k < netPtr->layers[i-1].numNodes; k++)
            {
                netPtr->layers[i].weightMatrix[j * netPtr->layers[i-1].numNodes + k] -= netPtr->layers[i].dCdZs[j] * netPtr->layers[i-1].activations[k] * learningRate;
            }
            netPtr->layers[i].biases[j] -= netPtr->layers[i].dCdZs[j] * learningRate;
        }
    }
}

float cost_function(NeuralNet* netPtr, float* expectedOutputBuffer)
{
    unsigned int i;
    unsigned int last = netPtr->numLayers - 1;
    float diff;
    float sum;
    sum = 0.0f;
    for (i = 0; i < netPtr->layers[last].numNodes; i++)
    {
        diff = netPtr->layers[last].activations[i] - expectedOutputBuffer[i];
        sum += diff * diff;
    }
    return sum / (netPtr->layers[last].numNodes);
}

void train(NeuralNet* netPtr, unsigned int numEpochs, float learningRate)
{
    unsigned int inputSize = netPtr->layers[0].numNodes;
    unsigned int outputSize = netPtr->layers[netPtr->numLayers - 1].numNodes;
    unsigned int epoch;
    unsigned int tPair;
    unsigned int rIndex;
    unsigned int i;
    float        cost;

    int dimX, dimY, numChannels;
    unsigned char* rgbData = stbi_load("1078.jpg", &dimX, &dimY, &numChannels, 0);
    
    float* expectedBufferPtr = malloc(500 * 500 * 3 * sizeof(float));
    for (i = 0; i < 500 * 500 * 3; i++)
    {
        expectedBufferPtr[i] = rgbData[i] / 255.0f;
    }
    
    for (i = 0; i < netPtr->layers[0].numNodes; i++)
    {
        netPtr->layers[0].activations[i] = random_float(0.0f, 1.0f);
    }

    for (epoch = 0; epoch < numEpochs; epoch++)
    {

        propagate_forward(netPtr);
        propagate_backward(netPtr, expectedBufferPtr, learningRate);
        cost = cost_function(netPtr, expectedBufferPtr);

        printf("Epoch %u complete. avgCost = %f\n", epoch+1, cost);
    }

    free(rgbData);
    free(expectedBufferPtr);
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

void gen_image(NeuralNet* netPtr, char* outputPath)
{
    propagate_forward(netPtr);

    unsigned char* rgbData = malloc(500 * 500 * 3 * sizeof(unsigned char));
    for (unsigned int i = 0; i < netPtr->layers[netPtr->numLayers-1].numNodes; i++)
    {
        rgbData[i] = (unsigned char)(netPtr->layers[netPtr->numLayers-1].activations[i] * 255);
    }

    stbi_write_jpg(outputPath, 500, 500, 3, rgbData, 70);
    
    free(rgbData);
}

void prompt_save(NeuralNet* netPtr)
{
    printf("Aight save to disk as? ");
    char outName[30];
    scanf("%s", outName);
    save_network_to_disk(netPtr, outName);
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    if (argc > 1)
    {
        printf("Loading %s\n", argv[1]);
        NeuralNet* netPtr = load_network_from_disk(argv[1]);
        
        if (!strcmp(argv[2], "-gen"))
        {
            gen_image(netPtr, argv[3]);
        }

        else if (!strcmp(argv[2], "-train"))
        {
            train(netPtr, atoi(argv[3]), atof(argv[4]));
            prompt_save(netPtr);
        }
    }
    else
    {   
        printf("Generating network with default settings\n");

        unsigned int numLayers = 4;
        unsigned int layerHeights[] = {100, 16, 16, 500 * 500 * 3};
        NeuralNet* netPtr = gen_neural_net(numLayers, layerHeights);

        printf("Randomizing network\n");
        randomize_network(netPtr);

        prompt_save(netPtr);
    }
    
    return 0;
}