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

#define THREAD_X 20
#define THREAD_Y 20

__device__ int counter = 0;

// Returns the index into the 1d pixel array
// Given te desired x,y, and image width
int pixelIndex(int x, int y, int width)
{
    return (y*width + x);
}

// Returns the sobel value for pixel x,y
int sobel(int x, int y, int width, char *pixels)
{
   int x00 = -1;  int x20 = 1;
   int x01 = -2;  int x21 = 2;
   int x02 = -1;  int x22 = 1;
   x00 *= pixels[pixelIndex(x-1,y-1,width)];
   x01 *= pixels[pixelIndex(x-1,y,width)];
   x02 *= pixels[pixelIndex(x-1,y+1,width)];
   x20 *= pixels[pixelIndex(x+1,y-1,width)];
   x21 *= pixels[pixelIndex(x+1,y,width)];
   x22 *= pixels[pixelIndex(x+1,y+1,width)];
   
   int y00 = -1;  int y10 = -2;  int y20 = -1;
   int y02 = 1;  int y12 = 2;  int y22 = 1;
   y00 *= pixels[pixelIndex(x-1,y-1,width)];
   y10 *= pixels[pixelIndex(x,y-1,width)];
   y20 *= pixels[pixelIndex(x+1,y-1,width)];
   y02 *= pixels[pixelIndex(x-1,y+1,width)];
   y12 *= pixels[pixelIndex(x,y+1,width)];
   y22 *= pixels[pixelIndex(x+1,y+1,width)];

   int px = x00 + x01 + x02 + x20 + x21 + x22;
   int py = y00 + y10 + y20 + y02 + y12 + y22;
   return sqrt(px*px + py*py);
}

__global__ void test() {
	printf("%d, %d (%d, %d): Hello!\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

__global__ void GPU_sobel(int width, char *pixels, RGBQUAD *returnPixels) {
    RGBQUAD aPixel;
    int threadsPerBlock = THREAD_X * THREAD_Y;
    int blockID = blockIdx.y * gridDim.x + blockIdx.x;
    int threadID = threadIdx.y * THREAD_X + threadIdx.x;
    int pixelID = blockID * threadsPerBlock + threadID;
    //printf("blockID(%d, %d, dimx: %d): %d, %d, %d\n", blockIdx.y, blockIdx.x, gridDim.x, blockID, threadID, pixelID);
    aPixel.rgbRed = 0;
    aPixel.rgbGreen = 0;
    aPixel.rgbBlue = 0;

    // if(0 < blockIdx.y && blockIdx.y < 11)
    // {
    //     aPixel.rgbRed = 255;
    // }

    if(blockIdx.y == 0)
    {
        aPixel.rgbRed = 255;
    }
    if(blockIdx.y == 1)
    {
        aPixel.rgbGreen = 255;
    }
    if(blockIdx.y == 2)
    {
        aPixel.rgbGreen = 20;
    }
    if(blockIdx.x == 0)
    {
        aPixel.rgbBlue = 255;
    }
    if(blockIdx.x == 1)
    {
        aPixel.rgbRed = 111;
    }
    if(blockIdx.y == 3)
    {
        aPixel.rgbGreen = 80;
    }
    //printf("(%d * %d + %d)(%d)+(%d * %d + %d) = %d\n", blockIdx.y, blockDim.x, blockIdx.x, threadsPerBlock, threadIdx.y, THREAD_X, threadIdx.x, pixelID);

    returnPixels[pixelID] = aPixel;
}

__global__ void GPU_grayscale(FIBITMAP *image, char *pixels) {
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
	
	int totalSobelPixels = (imgWidth) * (imgHeight);
    int blocksRequired = ceil(totalSobelPixels / (THREAD_X * THREAD_Y));
    int BLOCK_X = 10;
	int BLOCK_Y = ceil(blocksRequired/BLOCK_X);
	printf("%d rows of 512, %d is blocksRequired, %d is totalSobelPixels\n", BLOCK_Y,blocksRequired, totalSobelPixels);
	
	BLOCK_X = 3;
	BLOCK_Y = 3;
	
	dim3 threadsPerBlock(THREAD_X, THREAD_Y);
	dim3 numberOfBlocks(BLOCK_X, BLOCK_Y);
	// test<<<numberOfBlocks, threadsPerBlock>>>();

    // Convert image into a flat array of chars with the value 0-255 of the
    // greyscale intensity
	
    char *hostPixels;
	char *devicePixels;
	
	RGBQUAD aPixel;
	RGBQUAD *hostReturnPixels;
	RGBQUAD *deviceReturnPixels;
	
    int pixIndex = 0;
    int byteTotalPixels = sizeof(char)*imgWidth*imgHeight;
    
    hostPixels = (char *) malloc(byteTotalPixels);
    hostReturnPixels = (RGBQUAD *) malloc(byteTotalPixels);
    
	cudaMalloc((void**)&devicePixels, byteTotalPixels);
	cudaMalloc((void**)&deviceReturnPixels, byteTotalPixels);
	
    for (int i = 0; i < imgHeight; i++)
    {
        for (int j = 0; j < imgWidth; j++)
        {
            // FreeImage_GetPixelColor(image,j,i,&aPixel);
            // char grey = ((aPixel.rgbRed + aPixel.rgbGreen + aPixel.rgbBlue)/3);
            // hostPixels[pixIndex++]=grey;
        }
    }
	 
	cudaMemcpy(devicePixels, hostPixels, byteTotalPixels, cudaMemcpyHostToDevice);

	printf("Begin sobel...\n");
	
	GPU_sobel<<<numberOfBlocks, threadsPerBlock>>>(imgWidth, devicePixels, deviceReturnPixels);
	
	cudaMemcpy(hostPixels, devicePixels, byteTotalPixels, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostReturnPixels, deviceReturnPixels, byteTotalPixels, cudaMemcpyDeviceToHost);
	
    int pixelsCalculatedX_W = THREAD_X * BLOCK_X;
    int pixelsCalculatedY_H = THREAD_Y * BLOCK_Y;

	FIBITMAP *bitmap = FreeImage_Allocate(pixelsCalculatedX_W,pixelsCalculatedY_H, 24);
    // for (int i = 1; i < imgWidth-1; i++)
    // {
    //   for (int j = 1; j < imgHeight-1; j++)
    //   {
    for (int i = 0; i < pixelsCalculatedX_W; i++)
    {
      for (int j = 0; j < pixelsCalculatedY_H; j++)
      {
	    //int sVal = sobel(i,j,imgWidth,hostPixels);
	    //aPixel.rgbRed = sVal;
	    //aPixel.rgbGreen = sVal;
	    //aPixel.rgbBlue = sVal;
        FreeImage_SetPixelColor(bitmap, i, j, &hostReturnPixels[i + j *  pixelsCalculatedX_W]);
      }
    }
	printf("End sobel.\n");
    FreeImage_Save(FIF_PNG, bitmap, "coins-edge.png", 0);
  
    free(hostPixels);
	free(hostReturnPixels);
	cudaFree(devicePixels);
	cudaFree(deviceReturnPixels);
    FreeImage_Unload(bitmap);
    FreeImage_Unload(image);
    return 0;
}
