/***********************************************************************
 * See https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
 * or https://blog.saush.com/2011/04/20/edge-detection-with-the-sobel-operator-in-ruby/
 * for info on how the filter is implemented. */
 
 
 // Jason Hsi
 // System Architecture
 // HW4 Problem 1
 // sobel-gpu.cu
 
#include "FreeImage.h"
#include "stdio.h"
#include "math.h"

#define THREAD_XY_SIDE_DIM 22

// Returns the index into the 1d pixel array
// Given te desired x,y, and image width
__device__ int pixelIndex(int x, int y, int width)
{
    return (y*width + x);
}

// Returns the sobel value for pixel x,y
__global__ void sobel(int width, char *incomingPixels, char *returnPixels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int x00 = -1;  int x20 = 1;
    int x01 = -2;  int x21 = 2;
    int x02 = -1;  int x22 = 1;
    x00 *= incomingPixels[pixelIndex(x-1,y-1,width)];
    x01 *= incomingPixels[pixelIndex(x-1,y,width)];
    x02 *= incomingPixels[pixelIndex(x-1,y+1,width)];
    x20 *= incomingPixels[pixelIndex(x+1,y-1,width)];
    x21 *= incomingPixels[pixelIndex(x+1,y,width)];
    x22 *= incomingPixels[pixelIndex(x+1,y+1,width)];

    int y00 = -1;  int y10 = -2;  int y20 = -1;
    int y02 = 1;  int y12 = 2;  int y22 = 1;
    y00 *= incomingPixels[pixelIndex(x-1,y-1,width)];
    y10 *= incomingPixels[pixelIndex(x,y-1,width)];
    y20 *= incomingPixels[pixelIndex(x+1,y-1,width)];
    y02 *= incomingPixels[pixelIndex(x-1,y+1,width)];
    y12 *= incomingPixels[pixelIndex(x,y+1,width)];
    y22 *= incomingPixels[pixelIndex(x+1,y+1,width)];

    int px = x00 + x01 + x02 + x20 + x21 + x22;
    int py = y00 + y10 + y20 + y02 + y12 + y22;
    returnPixels[pixelIndex(x,y,width)] = sqrt(float(px*px + py*py));
}

int main()
{
    FreeImage_Initialise();
    atexit(FreeImage_DeInitialise);

    // Load image and get the width and height
    FIBITMAP *image;
    image = FreeImage_Load(FIF_PNG, "coins.png", 0);
    if (image == NULL)
    {
        printf("Image Load Problem\n");
        exit(0);
    }

    int imgWidth;
    int imgHeight;
    imgWidth = FreeImage_GetWidth(image);
    imgHeight = FreeImage_GetHeight(image);
	
    RGBQUAD aPixel;
    char *hostPixels;
    int totalPixelBytes = sizeof(char)*imgWidth*imgHeight;
    hostPixels = (char *) malloc(totalPixelBytes);
    int pixIndex = 0;
    
    for (int i = 0; i < imgHeight; i++)
    {
        for (int j = 0; j < imgWidth; j++)
        {
            FreeImage_GetPixelColor(image,j,i,&aPixel);
            char grey = ((aPixel.rgbRed + aPixel.rgbGreen + aPixel.rgbBlue)/3);
            hostPixels[pixIndex++]=grey;
        }
    }

    int BLOCK_X = ceil(imgWidth/THREAD_XY_SIDE_DIM);
    int BLOCK_Y = ceil(imgHeight/THREAD_XY_SIDE_DIM);
	
	dim3 threadsPerBlock(THREAD_XY_SIDE_DIM, THREAD_XY_SIDE_DIM, 1);
    dim3 numberOfBlocks(BLOCK_X, BLOCK_Y, 1);
    
    // Convert image into a flat array of chars with the value 0-255 of the
    // greyscale intensity
    
    char *devicePixels;
    char *deviceReturnPixels;
    char *hostReturnPixels;
    hostReturnPixels = (char *) malloc(totalPixelBytes);
	cudaMalloc((void**)&devicePixels, totalPixelBytes);
	cudaMalloc((void**)&deviceReturnPixels, totalPixelBytes);
    cudaMemcpy(devicePixels, hostPixels, totalPixelBytes, cudaMemcpyHostToDevice);
	sobel<<<numberOfBlocks, threadsPerBlock>>>(imgWidth, devicePixels, deviceReturnPixels);
    cudaMemcpy(hostReturnPixels, deviceReturnPixels, totalPixelBytes, cudaMemcpyDeviceToHost);
    
	FIBITMAP *bitmap = FreeImage_Allocate(imgWidth, imgHeight, 24);
    for (int i = 1; i < imgWidth-1; i++)
    {
      for (int j = 1; j < imgHeight-1; j++)
      {
	    int sVal = float(hostReturnPixels[j * imgWidth + i]);
	    aPixel.rgbRed = sVal;
	    aPixel.rgbGreen = sVal;
	    aPixel.rgbBlue = sVal;
        FreeImage_SetPixelColor(bitmap, i, j, &aPixel);
      }
    }
    FreeImage_Save(FIF_PNG, bitmap, "coins-edge.png", 0);

    free(hostPixels);
	free(hostReturnPixels);
	cudaFree(devicePixels);
    cudaFree(deviceReturnPixels);
    FreeImage_Unload(bitmap);
    FreeImage_Unload(image);

    return 0;
}
