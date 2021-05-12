/*
This neural network implementation uses a slightly different backpropagation algorithm in which the dCdw's and dCdb's are summed over every training pair in each epoch
and then at the end of the epoch are multiplied by the learning rate, divided by the number of training pairs in the batch and then subtracted from their associated weight/bias.
For being more complicated, this algorithm seems also to be pretty bad in comparison to directly modifying the weights and biases on each round of forward+backward propagation, which is how
vectorized_nn.c works.
*/

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
    float* biasNudges;
    float* weightMatrix;
    float* weightNudges;
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
        netPtr->layers[i].biasNudges = malloc(layerHeights[i] * sizeof(float));
        netPtr->layers[i].dCdZs = malloc(layerHeights[i] * sizeof(float));
        netPtr->layers[i].weightMatrix = malloc(layerHeights[i] * layerHeights[i-1] * sizeof(float));
        netPtr->layers[i].weightNudges = malloc(layerHeights[i] * layerHeights[i-1] * sizeof(float));
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

    for (i = last; i > 0; i--)      //now calculate and add deltas to biasNudges and weightNudges
    {
        for (j = 0; j < netPtr->layers[i].numNodes; j++)
        {
            for (k = 0; k < netPtr->layers[i-1].numNodes; k++)
            {
                netPtr->layers[i].weightNudges[j * netPtr->layers[i-1].numNodes + k] += netPtr->layers[i].dCdZs[j] * netPtr->layers[i-1].activations[k];
            }
            netPtr->layers[i].biasNudges[j] += netPtr->layers[i].dCdZs[j];
        }
    }
}

float apply_nudges(NeuralNet* netPtr, float learningRate, unsigned int batchSize)
{
    unsigned int i;
    unsigned int j;
    unsigned int k;

    unsigned int last = netPtr->numLayers-1;

    for (i = last; i > 0; i--)
    {
        for (k = 0; k < netPtr->layers[i].weightMatrixSize; k++)
        {
            netPtr->layers[i].weightMatrix[k] -= netPtr->layers[i].weightNudges[k] * learningRate / (float)batchSize;
        }
        for (j = 0; j < netPtr->layers[i].numNodes; j++)
        {
            netPtr->layers[i].biases[j] -= netPtr->layers[i].biasNudges[j] * learningRate / (float)batchSize;
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
    return sum;
}

void reset_nudges(NeuralNet* netPtr)
{
    unsigned int i;
    unsigned int j;
    for (i = 1; i < netPtr->numLayers; i++)
    {
        for (j = 0; j < netPtr->layers[i].numNodes; j++)
        {
            netPtr->layers[i].biasNudges[j] = 0.0f;
        }
        for (j = 0; j < netPtr->layers[i].weightMatrixSize; j++)
        {
            netPtr->layers[i].weightNudges[j] = 0.0f;
        }
    }
}

void train(NeuralNet* netPtr, TrainingData* tDataPtr, unsigned int batchSize, unsigned int numEpochs, float learningRate)
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
        reset_nudges(netPtr);
        for (tPair = 0; tPair < batchSize; tPair++)
        {
            rIndex = random_uint(tDataPtr->numPairs);
            memcpy(netPtr->layers[0].activations, tDataPtr->inputData + (rIndex * inputSize), inputSize * sizeof(float));
            propagate_forward(netPtr);
            propagate_backward(netPtr, tDataPtr->outputData + (rIndex * outputSize));
            sumCost += cost_function(netPtr, tDataPtr->outputData + (rIndex * outputSize));
        }
        apply_nudges(netPtr, learningRate, batchSize);
        printf("Epoch %u complete. avgCost = %f\n", epoch+1, sumCost / batchSize);
    }
}

TrainingData* load_training_data(char* filePathInputs, char* filePathOutputs, unsigned int numPairs, unsigned int inputSize, unsigned int outputSize, unsigned int inputFileByteOffset, unsigned int outputFileByteOffset, char* inputType, char* outputType, char* outputMethod)
{
    TrainingData* tDataPtr = malloc(sizeof(TrainingData));
    tDataPtr->numPairs = numPairs;
    tDataPtr->inputSize = inputSize;
    tDataPtr->inputData = malloc(numPairs * inputSize * sizeof(float));
    tDataPtr->outputSize = outputSize;
    tDataPtr->outputData = malloc(numPairs * outputSize * sizeof(float));

    FILE* fptrI = fopen(filePathInputs, "rb");
    fseek(fptrI, inputFileByteOffset, SEEK_SET);
    FILE* fptrO = fopen(filePathOutputs, "rb");
    fseek(fptrO, outputFileByteOffset, SEEK_SET);

    unsigned int i;

    if (!strcmp(inputType, "float"))        //First the inputs
    {
        fread(tDataPtr->inputData, numPairs * inputSize * sizeof(float), 1, fptrI);
    }
    else if (!strcmp(inputType, "char"))
    {
        for (i = 0; i < numPairs * inputSize; i++)
        {
            tDataPtr->inputData[i] = (float)(fgetc(fptrI)) / 255.0f;
        }
    }
    else
    {
        printf("Error in load_training_data: Invalid inputType\n");
        goto EXIT;
    }

    if (!strcmp(outputMethod, "highest-node"))
    {
        if (!strcmp(outputType, "char"))
        {
            for (i = 0; i < numPairs * outputSize; i++)
            {
                tDataPtr->outputData[i] = 0.0f;
            }
            for (i = 0; i < numPairs * outputSize; i += outputSize)
            {
                tDataPtr->outputData[i + fgetc(fptrO)] = 1.0f;
            }
        }
    }
    else
    {
        printf("Error in load_training_data: Invalid outputMethod\n");
        goto EXIT;
    }

    fclose(fptrI);
    fclose(fptrO);
    return tDataPtr;
    EXIT: {
        fclose(fptrI);
        fclose(fptrO);
        exit(1);
    }
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
        netPtr->layers[i].biasNudges = malloc(netPtr->layers[i].numNodes * sizeof(float));
        netPtr->layers[i].dCdZs = malloc(netPtr->layers[i].numNodes * sizeof(float));
        netPtr->layers[i].weightMatrix = malloc(netPtr->layers[i].weightMatrixSize * sizeof(float));
        netPtr->layers[i].weightNudges = malloc(netPtr->layers[i].weightMatrixSize * sizeof(float));

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
    printf("Done. Generating neural network...\n");

    NeuralNet* netPtr;
    if (argc > 1)
    {
        netPtr = load_network_from_disk(argv[1]);
    }
    else
    {
        unsigned int layerHeights[4] = {28*28, 16, 16, 10};
        netPtr = gen_neural_net(4, layerHeights);
        printf("Randomizing weights and biases...\n");
        randomize_network(netPtr);
    }

    printf("Done.\n");

    printf("training...\n");
    train(netPtr, tDataPtr, 10000, 100, 1.25f);
    printf("Saving to disk as joj.network\n");
    save_network_to_disk(netPtr, "joj.network");
    printf("Done.\n");
    test_network(netPtr, tDataPtr, 0, 50);
}