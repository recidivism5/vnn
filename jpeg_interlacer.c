#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

unsigned char* interlace(unsigned char* a, unsigned char* b, unsigned int dimX, unsigned int dimY, unsigned int numChannels)
{
    size_t size = dimX * dimY * numChannels * sizeof(unsigned char);
    unsigned char* output = malloc(size);
    unsigned int i;
    size_t lineBytes = dimX * numChannels * sizeof(unsigned char);
    memcpy(output, a, size);
    for (i = 0; i < dimY; i+=2)
    {
        memcpy(output+i*dimX*numChannels, b+i*dimX*numChannels, lineBytes);
    }
    return output;
}

int main(unsigned int argc, char** argv)
{
    if (argc < 4)
    {
        printf("Usage: jpeg_interlacer.exe img1 img2 outputName");
        exit(0);
    }
    unsigned int dimX, dimY, numChannels;
    unsigned int dim2X, dim2Y;
    unsigned char* img1Data = stbi_load(argv[1], &dimX, &dimY, &numChannels, 0);
    unsigned char* img2Data = stbi_load(argv[2], &dim2X, &dim2Y, &numChannels, 0);
    if ((dimX != dim2X) || (dimY != dim2Y))
    {
        printf("Images are not of same dimension");
        exit(0);
    }
    unsigned char* outputData = interlace(img1Data, img2Data, dimX, dimY, numChannels);
    stbi_write_jpg(argv[3], dimX, dimY, 3, outputData, 70);
}