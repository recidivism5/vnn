#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

float learningRate = 0.05f;

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
        netPtr->layers[i].weightMatrix = malloc(layerHeights[i] * layerHeights[i-1] * sizeof(float));
    }
    return netPtr;
}

void randomize_network(NeuralNet* netPtr)
{
    unsigned int i;
    unsigned int j;
    unsigned int k;

    for (i = 0; i < netPtr->numLayers; i++)
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

void propagate_backward(NeuralNet* netPtr, float* expectedOutputBuffer)
{
    unsigned int i;
    unsigned int j;
    unsigned int k;

    float dCda;

    unsigned int last;
    last = netPtr->numLayers - 1;
    for (i = 0; i < netPtr->layers[last].numNodes; i++)     //get dCdZs for output layer
    {
        netPtr->layers[last].dCdZs[i] = 2 * (netPtr->layers[last].activations[i] - expectedOutputBuffer[i]) * fast_sigmoid_derivative(netPtr->layers[last].activations[i]);
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
        sum += diff;
    }
    return sum / netPtr->layers[last].numNodes;
}

void train(NeuralNet* netPtr, float* inputData, float* outputData, unsigned int numTrainingPairs, unsigned int batchSize, unsigned int numEpochs)
{
    unsigned int inputSize = netPtr->layers[0].numNodes;
    unsigned int outputSize = netPtr->layers[netPtr->numLayers - 1].numNodes;
    unsigned int epoch;
    unsigned int tPair;
    unsigned int rIndex;
    float        sumCost;

    for (epoch = 0; epoch < numEpochs; epoch++)
    {
        sumCost = 0.0f;
        for (tPair = 0; tPair < batchSize; tPair++)
        {
            rIndex = random_uint(numTrainingPairs);
            memcpy(netPtr->layers[0].activations, inputData + (rIndex * inputSize), inputSize * sizeof(float));
            propagate_forward(netPtr);
            propagate_backward(netPtr, outputData + (rIndex * outputSize));
            sumCost += cost_function(netPtr, outputData + (rIndex * outputSize));
        }
        printf("Epoch %u complete. avgCost = %f", epoch+1, sumCost / batchSize);
    }
}

int main(int argc, char** argv)
{
    srand(time(NULL));

}