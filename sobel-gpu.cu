a/***********************************************************************
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

#define THREAD_X 16
#define THREAD_Y 16

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

__global__ void GPU_sobel(int x, int y, int width, char *pixels, double *returnVal) {

}

__global__ void GPU_grayscale(FIBITMAP &image, char *pixels) {
	RGBQUAD aPixel;
	FreeImage_GetPixelColor(image,0,0,&aPixel);
	char grey = ((aPixel.rgbRed + aPixel.rgbGreen + aPixel.rgbBlue)/3);
    pixels[pixIndex++]=grey;
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
	
	int totalPixels = imgWidth * imgHeight;
	int blocksRequired = ceil(imgWidth * imgHeight / (THREAD_X * THREAD_Y));
	int block2DDim = ceil(sqrt(blocksRequired));
	printf("%d is block 2D dim, %d is blocksRequired, %d is totalPixels\n", block2DDim, blocksRequired, totalPixels);
	
	const int BLOCK_X = block2DDim;
	const int BLOCK_Y = block2DDim;
	
	dim3 threadsPerBlock(THREAD_X, THREAD_Y);
	dim3 numberOfBlocks(BLOCK_X, BLOCK_Y);
	// test<<<numberOfBlocks, threadsPerBlock>>>();

    // Convert image into a flat array of chars with the value 0-255 of the
    // greyscale intensity
	
    char *hostPixels;
	char *devicePixels;
	
    int pixIndex = 0;
    hostPixels = (char *) malloc(sizeof(char)*imgWidth*imgHeight);
	cudaMalloc((void**)&devicePixels, sizeof(char)*imgWidth*imgHeight);
    for (int i = 0; i < imgHeight; i++)
     for (int j = 0; j < imgWidth; j++)
     {
       FreeImage_GetPixelColor(image,j,i,&aPixel);
       char grey = ((aPixel.rgbRed + aPixel.rgbGreen + aPixel.rgbBlue)/3);
       hostPixels[pixIndex++]=grey;
     }

    // Apply sobel operator to hostPixels, ignoring the borders
    FIBITMAP *bitmap = FreeImage_Allocate(imgWidth, imgHeight, 24);
    for (int i = 1; i < imgWidth-1; i++)
    {
      for (int j = 1; j < imgHeight-1; j++)
      {
	    int sVal = sobel(i,j,imgWidth,hostPixels);
	    aPixel.rgbRed = 0;
	    aPixel.rgbGreen = 0;
	    aPixel.rgbBlue = sVal;
        FreeImage_SetPixelColor(bitmap, i, j, &aPixel);
      }
    }
    FreeImage_Save(FIF_PNG, bitmap, "coins-edge.png", 0);
  
    free(hostPixels);
    FreeImage_Unload(bitmap);
    FreeImage_Unload(image);
    return 0;
}
