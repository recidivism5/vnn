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

    printf("discriminator output = %f\nexpected output = %f\n", netPtr->layers[netPtr->numLayers-1].activations[0], expectedOutputBuffer[0]);

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

void propagate_backward_bridge(NeuralNet* net0Ptr, NeuralNet* net1Ptr)
{
    unsigned int i;
    unsigned int j;
    unsigned int k;

    float dCda;

    //fill dCdZs on output layer of net0
    unsigned int last;
    last = net0Ptr->numLayers-1;
    for (j = 0; j < net0Ptr->layers[last].numNodes; j++)
    {
        dCda = 0.0f;
        for (k = 0; k < net1Ptr->layers[1].numNodes; k++)
        {
            dCda += net1Ptr->layers[1].dCdZs[k] * net1Ptr->layers[1].weightMatrix[k * net0Ptr->layers[last].numNodes + j];
        }
        net0Ptr->layers[last].dCdZs[j] = dCda * fast_sigmoid_derivative(net0Ptr->layers[last].activations[j]);
    }

    for (i = last - 1; i > 0; i--)      //now get dCdZs for hidden layers of net0
    {
        for (j = 0; j < net0Ptr->layers[i].numNodes; j++)
        {
            dCda = 0.0f;
            for (k = 0; k < net0Ptr->layers[i+1].numNodes; k++)
            {
                dCda += net0Ptr->layers[i+1].dCdZs[k] * net0Ptr->layers[i+1].weightMatrix[k * net0Ptr->layers[i].numNodes + j];
            }
            net0Ptr->layers[i].dCdZs[j] = dCda * fast_sigmoid_derivative(net0Ptr->layers[i].activations[j]);
        }
    }

    for (i = last; i > 0; i--)      //now calculate and add deltas to biasNudges and weightNudges of net0
    {
        for (j = 0; j < net0Ptr->layers[i].numNodes; j++)
        {
            for (k = 0; k < net0Ptr->layers[i-1].numNodes; k++)
            {
                net0Ptr->layers[i].weightNudges[j * net0Ptr->layers[i-1].numNodes + k] += net0Ptr->layers[i].dCdZs[j] * net0Ptr->layers[i-1].activations[k];
            }
            net0Ptr->layers[i].biasNudges[j] += net0Ptr->layers[i].dCdZs[j];
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
            netPtr->layers[i].weightMatrix[k] -= netPtr->layers[i].weightNudges[k] * learningRate;// / (float)batchSize;
        }
        for (j = 0; j < netPtr->layers[i].numNodes; j++)
        {
            netPtr->layers[i].biases[j] -= netPtr->layers[i].biasNudges[j] * learningRate;// / (float)batchSize;
        }
    }
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
        for (tPair = 0; tPair < batchSize; tPair++)
        {
            rIndex = random_uint(tDataPtr->numPairs);
            memcpy(netPtr->layers[0].activations, tDataPtr->inputData + (rIndex * inputSize), inputSize * sizeof(float));
            propagate_forward(netPtr);
            propagate_backward(netPtr, tDataPtr->outputData + (rIndex * outputSize));
            sumCost += cost_function(netPtr, tDataPtr->outputData + (rIndex * outputSize));
        }
        printf("Epoch %u complete. avgCost = %f\n", epoch+1, sumCost / batchSize);
    }
}

void train_gan(NeuralNet* genPtr, NeuralNet* discPtr, unsigned int numImages, unsigned int numRounds, float learningRate)
{
    unsigned int imageId;
    char path[30];
    strcpy(path, "./sg_2/");
    int dimX, dimY, numChannels;
    unsigned char* rgbData;
    char imgOutputPath[30];
    float expected;
    unsigned int r;
    unsigned int i;
    for (r = 0; r < numRounds; r++)
    {
        reset_nudges(genPtr);
        reset_nudges(discPtr);

        imageId = random_uint(numImages);
        sprintf(path+7, "%d", imageId);
        strcat(path, ".jpg");
        //printf("%s\n", path);
        rgbData = stbi_load(path, &dimX, &dimY, &numChannels, 0);
        //printf("x: %i y: %i numChannels: %i\n", dimX, dimY, numChannels);
        //stbi_write_jpg("joj.jpg", 100, 100, 3, rgbData, 50);
        
        //train discriminator on real image and fake image:
        for (i = 0; i < discPtr->layers[0].numNodes; i++)
        {
            discPtr->layers[0].activations[i] = rgbData[i] / 255.0f;
        }
        propagate_forward(discPtr);
        expected = 1.0f;
        propagate_backward(discPtr, &expected);

        for (i = 0; i < genPtr->layers[0].numNodes; i++)
        {
            genPtr->layers[0].activations[i] = random_float(0.0f, 1.0f);
        }
        propagate_forward(genPtr);

        if (!(r % 100))
        {
            printf("Saving generator output to %i.jpg\n", r);
            sprintf(imgOutputPath, "%d", r);
            strcat(imgOutputPath, ".jpg");
            //reuse rgbData:
            for (i = 0; i < genPtr->layers[genPtr->numLayers-1].numNodes; i++)
            {
                rgbData[i] = (unsigned char)(genPtr->layers[genPtr->numLayers-1].activations[i] * 255);
            }
            stbi_write_jpg(imgOutputPath, 100, 100, 3, rgbData, 70);
        }

        memcpy(discPtr->layers[0].activations, genPtr->layers[genPtr->numLayers-1].activations, discPtr->layers[0].numNodes * sizeof(float));
        propagate_forward(discPtr);
        expected = 0.0f;
        propagate_backward(discPtr, &expected);

        apply_nudges(discPtr, learningRate, 2);
        reset_nudges(discPtr);

        //now train generator
        expected = 1.0f;
        propagate_backward(discPtr, &expected);
        propagate_backward_bridge(genPtr, discPtr);
        apply_nudges(genPtr, learningRate, 1);

        free(rgbData);
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
    
    //train_gan(0, 0, 1000, 10, 0.5f);

    //make generator network
    unsigned int LH1[4] = {100, 16, 16, 100 * 100 * 3};
    NeuralNet* generatorPtr = gen_neural_net(4, LH1);
    randomize_network(generatorPtr);
    //make discriminator network
    unsigned int LH2[4] = {100 * 100 * 3, 16, 16, 1};
    NeuralNet* discriminatorPtr = gen_neural_net(4, LH2);
    randomize_network(discriminatorPtr);
    printf("GAN INITIALIZED\n");
    train_gan(generatorPtr, discriminatorPtr, 1000, 50000, 0.0002f);
}