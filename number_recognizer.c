#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define LEARNING_RATE 0.05f

int isLittleEndian()
{
    short int number = 0x1;
    char *numPtr = (char*)&number;
    return (numPtr[0] == 1);
}

float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float fast_sigmoid_derivative(float fx)
{
    return fx * (1-fx);
}

float average(float* ar, unsigned int numIndices)
{
    float output = 0;
    unsigned int i;
    for (i = 0; i < numIndices; i++)
    {
        output += ar[i];
    }
    return output / numIndices;
}

float random_float(float Min, float Max)
{
    return (((float)rand() / (float)RAND_MAX) * (Max - Min)) + Min;
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

typedef struct TrainingPair {
    float* inputs;
    float* expectedActivation;
} TrainingPair;

typedef struct TrainingData {
    TrainingPair* pairs;
    unsigned int numPairs;
    unsigned int inputLength;
    unsigned int outputLength;
} TrainingData;

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

float errorSum(Layer* outputLayer, float* trainingBuffer)
{
    float sum = 0;
    unsigned int i;
    for (i = 0; i < outputLayer->numNeurons; i++)
    {
        sum += outputLayer->neurons[i].activation - trainingBuffer[i];
    }
    return sum;
}

void print_output_layer_activations(NeuralNet* netPtr)
{
    unsigned int i;
    for (i = 0; i < netPtr->layers[netPtr->numLayers-1].numNeurons; i++)
    {
        printf("%f ", netPtr->layers[netPtr->numLayers-1].neurons[i].activation);
    }
    putchar('\n');
}

void train(NeuralNet* netPtr, TrainingData* tDataPtr, unsigned int cycles)
{
    float sumCost;
    
    float dCda;

    unsigned int cycle;
    unsigned int i;
    unsigned int j;
    unsigned int k;
    unsigned int w;
    unsigned int n;
    for (cycle = 0; cycle < cycles; cycle++)
    {
        sumCost = 0;
        for (i = 0; i < tDataPtr->numPairs; i++) //foreach training pair
        {
            for (j = 0; j < tDataPtr->inputLength; j++)
            {
                netPtr->layers[0].neurons[j].activation = tDataPtr->pairs[i].inputs[j];
            }
            for (j = 0; j < netPtr->numLayers - 1; j++)
            {
                calculate_activations(&(netPtr->layers[j]), &(netPtr->layers[j+1]));
            }
            sumCost += cost_function(&(netPtr->layers[netPtr->numLayers-1]), tDataPtr->pairs[i].expectedActivation);
            //forward propagation done, now backpropagation:


            for (j = netPtr->numLayers-1; j > 0; j--)
            {
                if (j == netPtr->numLayers-1)
                {
                    for (k = 0; k < netPtr->layers[j].numNeurons; k++)
                    {
                        netPtr->layers[j].neurons[k].dCdZ = 2 * (netPtr->layers[j].neurons[k].activation - tDataPtr->pairs[i].expectedActivation[k]) * fast_sigmoid_derivative(netPtr->layers[j].neurons[k].activation);
                    }
                }
                else
                {
                    for (k = 0; k < netPtr->layers[j].numNeurons; k++)
                    {
                        dCda = 0.0f;
                        for (n = 0; n < netPtr->layers[j+1].numNeurons; n++)
                        {
                            dCda += netPtr->layers[j+1].neurons[n].dCdZ * netPtr->layers[j+1].neurons[n].weights[k];
                        }
                        netPtr->layers[j].neurons[k].dCdZ = dCda * fast_sigmoid_derivative(netPtr->layers[j].neurons[k].activation);
                    }
                }
                for (k = 0; k < netPtr->layers[j].numNeurons; k++)
                {
                    for (w = 0; w < netPtr->layers[j].neurons[k].numWeights; w++)
                    {
                        netPtr->layers[j].neurons[k].weights[w] -= (netPtr->layers[j].neurons[k].dCdZ * netPtr->layers[j-1].neurons[w].activation) * LEARNING_RATE;
                    }
                    netPtr->layers[j].neurons[k].bias -= netPtr->layers[j].neurons[k].dCdZ * LEARNING_RATE;
                }
            }
        }
        
        printf("Cycle %u complete. avgCost = %f\n", cycle, sumCost / tDataPtr->numPairs);
    }
}

TrainingData* load_training_data(char* imagesPath, char* labelsPath)
{
    unsigned int numItems = 60000;
    unsigned int imageSize = 28 * 28;
    unsigned int outputLength = 10;
    TrainingData* tDataPtr = malloc(sizeof(TrainingData));
    tDataPtr->numPairs = numItems;
    tDataPtr->inputLength = imageSize;
    tDataPtr->outputLength = outputLength;
    tDataPtr->pairs = malloc(numItems * sizeof(TrainingPair));

    FILE* fptrImgs = fopen(imagesPath, "rb");
    fseek(fptrImgs, 16, SEEK_SET);
    FILE* fptrLabels = fopen(labelsPath, "rb");
    fseek(fptrLabels, 8, SEEK_SET);
    unsigned int i;
    unsigned int j;
    unsigned int k;
    for (i = 0; i < numItems; i++)
    {
        tDataPtr->pairs[i].inputs = malloc(imageSize * sizeof(float));
        tDataPtr->pairs[i].expectedActivation = malloc(outputLength * sizeof(float));
        for (j = 0; j < imageSize; j++)
        {
            tDataPtr->pairs[i].inputs[j] = ((float)fgetc(fptrImgs)) / 255.0f;
        }
        for (k = 0; k < outputLength; k++)
        {
            tDataPtr->pairs[i].expectedActivation[k] = 0.0f;
        }
        tDataPtr->pairs[i].expectedActivation[fgetc(fptrLabels)] = 1.0f;
    }
    fclose(fptrImgs);
    fclose(fptrLabels);
    return tDataPtr;
}

NeuralNet* gen_network(unsigned int numLayers, unsigned int* layerHeights)
{
    NeuralNet* netPtr = malloc(sizeof(NeuralNet));
    netPtr->numLayers = numLayers;
    netPtr->layers = malloc(numLayers * sizeof(Layer));
    
    netPtr->layers[0].numNeurons = layerHeights[0];
    netPtr->layers[0].neurons = malloc(layerHeights[0] * sizeof(Neuron)); //skip weights and bias on input layer

    unsigned int i;
    unsigned int j;
    unsigned int k;
    for (i = 1; i < numLayers; i++)
    {
        netPtr->layers[i].numNeurons = layerHeights[i];
        netPtr->layers[i].neurons = malloc(layerHeights[i] * sizeof(Neuron));
        for (j = 0; j < layerHeights[i]; j++)
        {
            netPtr->layers[i].neurons[j].numWeights = netPtr->layers[i-1].numNeurons;
            netPtr->layers[i].neurons[j].weights = malloc(netPtr->layers[i-1].numNeurons * sizeof(float));
            for (k = 0; k < netPtr->layers[i].neurons[j].numWeights; k++)
            {
                netPtr->layers[i].neurons[j].weights[k] = random_float(-1.0f, 1.0f);
            }
            netPtr->layers[i].neurons[j].bias = random_float(-1.0f, 1.0f);
        }
    }
    return netPtr;
}

void save_model_to_disk(NeuralNet* netPtr, char* filePath)
{
    FILE* fptr = fopen(filePath, "wb");
    fwrite(&(netPtr->numLayers), sizeof(netPtr->numLayers), 1, fptr);
    unsigned int i;
    unsigned int j;
    unsigned int k;
    for (i = 0; i < netPtr->numLayers; i++)
    {
        fwrite(&(netPtr->layers[i].numNeurons), sizeof(netPtr->layers[i].numNeurons), 1, fptr);
    }
    for (i = 1; i < netPtr->numLayers; i++)
    {
        for (j = 0; j < netPtr->layers[i].numNeurons; j++)
        {
            fwrite(&(netPtr->layers[i].neurons[j].bias), sizeof(float), 1, fptr);
            fwrite(netPtr->layers[i].neurons[j].weights, netPtr->layers[i].neurons[j].numWeights * sizeof(float), 1, fptr);
        }
    }
    fclose(fptr);
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

int main(int argc, char* argv[])
{
    srand(time(NULL));

    printf("Loading training data... ");
    TrainingData* tDataPtr = load_training_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    printf("Done.\nGenerating neural net... ");

    NeuralNet* netPtr;
    if (argc < 4)
    {
        unsigned int layerHeights[] = {28*28, 16, 16, 10};
        netPtr = gen_network(4, layerHeights);
    }
    else
    {
        netPtr = load_network(argv[3]);
    }
    
    unsigned int cycles = 1000;
    if (argv[1] != NULL)
    {
        sscanf(argv[1], "%u", &cycles);
    }
    printf("Done.\nTraining for %u cycles...\n", cycles);
    train(netPtr, tDataPtr, cycles);
    printf("Done.\n");

    char saveName[30] = "0";
    if (argv[2] != NULL)
    {
        strcpy(&(saveName[0]), argv[2]);
    }
    strcat(saveName, ".network");
    printf("Saving model to disk as %s\n", saveName);
    save_model_to_disk(netPtr, saveName);
    printf("Done.\n");
}