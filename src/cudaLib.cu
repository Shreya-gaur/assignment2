
#include "cudaLib.cuh"
#include <thrust/device_vector.h>

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 3.14159f;

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}


#define BLOCK_SIZE 16
#define MAX_WINDOW_SIZE 64

void query_device(){

	int maxThreadPerBlock, maxBlockDim, maxGridDim;
	int maxSharedMemoryPerBlock, maxWarpSize;
	
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		// return -1;
	}
	else{

		cudaError_t err_ThreadPerBlock = cudaDeviceGetAttribute(&maxThreadPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
 		// printf("err = %d, %s\n", err_ThreadPerBlock, cudaGetErrorString(err_ThreadPerBlock));
  		printf("maxThreadPerBlock = %d\n", maxThreadPerBlock);
		
		cudaError_t err_BlockDim = cudaDeviceGetAttribute(&maxBlockDim, cudaDevAttrMaxBlockDimX, 0);
 		// printf("err = %d, %s\n", err_BlockDim, cudaGetErrorString(err_BlockDim));
  		printf("maxBlockDim = %d\n", maxBlockDim);

		cudaError_t err_sm = cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
 		// printf("err = %d, %s\n", err_BlockDim, cudaGetErrorString(err_BlockDim));
  		printf("maxSharedMemoryPerBlock = %d\n", maxSharedMemoryPerBlock);

		cudaError_t err_GridDim = cudaDeviceGetAttribute(&maxGridDim, cudaDevAttrMaxGridDimX, 0);
 		// printf("err = %d, %s\n", err_GridDim, cudaGetErrorString(err_GridDim));
  		printf("maxGridDim = %d\n", maxGridDim);

		cudaError_t err_WarpSize = cudaDeviceGetAttribute(&maxWarpSize, cudaDevAttrWarpSize, 0);
 		// printf("err = %d, %s\n", err_BlockDim, cudaGetErrorString(err_BlockDim));
  		printf("maxWarpSize = %d\n", maxWarpSize);

	}

}

__device__
uint8_t * sortPixels_gpu (uint8_t * array, dim3 arrayDim){

	uint8_t tmp;
	
	for (int i = 0; i < arrayDim.x - 1; i++) {
        for (int j = i+1; j < arrayDim.x; j++) {
            if (array[i] > array[j]) { 
				//Swap Values.
                tmp = array[i];
                array[i] = array[j];
                array[j] = tmp;
			}
		}
	}

	return array;
};


__global__ 
void medianFilter_gpu (uint8_t * inPixels, ImageDim imgDim, uint8_t * outPixels, MedianFilterArgs args) {

	int row_gl = blockDim.y * blockIdx.y + threadIdx.y;
	int col_gl = blockDim.x * blockIdx.x + threadIdx.x;
	int channels_gl = blockDim.z * blockIdx.z + threadIdx.z;

	int count = 0;

	uint32_t inRow, inCol;

	// __shared__ uint8_t window[MAX_WINDOW_SIZE][MAX_WINDOW_SIZE][3];
	uint8_t window[MAX_WINDOW_SIZE];
	uint8_t * sorted_window = window;

	if (col_gl < imgDim.width && row_gl < imgDim.height && channels_gl < 4) {

		for (uint32_t filRow = 0; filRow < args.filterH; ++ filRow) {
			for (uint32_t filCol = 0; filCol < args.filterW; ++ filCol) {
				inRow = row_gl + filRow;
				inCol = col_gl + filCol;
				
				if(inRow >= 0 && inRow < imgDim.height && 
					inCol >= 0 && inCol < imgDim.width){

					window[count] = inPixels[(inRow * imgDim.width + inCol) * imgDim.channels + channels_gl];
					count++;

				}
			}

		}

		dim3 arraydim(count);

		sorted_window = sortPixels_gpu(sorted_window, arraydim);

		outPixels[imgDim.channels*(row_gl* imgDim.width + col_gl) + channels_gl] = sorted_window[(args.filterH * args.filterW) / 2];

	}	
}


int runGpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {

	ImageDim imgDim;

	uint8_t * imgData, * imgData_d;

	int bytesRead = loadBytesImage(imgPath, imgDim, &imgData);
	int imgSize = imgDim.height * imgDim.width * imgDim.channels * imgDim.pixelSize;

	uint8_t * outData_d;
	uint8_t * outData = (uint8_t *) malloc(imgSize);

	cudaMalloc(&imgData_d, imgSize * sizeof(uint8_t));
	cudaMalloc(&outData_d, imgSize * sizeof(uint8_t));

	cudaMemcpy(imgData_d, imgData, imgSize * sizeof(uint8_t) , cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 3);
    dim3 dimGrid(ceil((float)imgDim.width / (float)dimBlock.x), ceil((float)imgDim.height / (float)dimBlock.y));
	medianFilter_gpu<<<dimGrid, dimBlock>>>(imgData_d, imgDim, outData_d, args);

	cudaMemcpy(outData, outData_d, imgSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	cudaFree(imgData_d);
	cudaFree(outData_d);

	writeBytesImage(outPath, imgDim, outData);

	return 0;
}


__global__ 
void poolLayer_gpu (float * input, TensorShape inShape, float * output, TensorShape outShape, PoolLayerArgs args){

	int row_gl = blockDim.y * blockIdx.y + threadIdx.y;
	int col_gl = blockDim.x * blockIdx.x + threadIdx.x;

	uint32_t row, col;

	if (col_gl < outShape.width && row_gl < outShape.height) {

		//	STUDENT: Assign to first value of pool area
		float poolPick = input[row_gl * args.strideH * inShape.width + col_gl * args.strideW];

		for (uint32_t poolRow = 0; poolRow < args.poolH; ++ poolRow) {
			for (uint32_t poolCol = 0; poolCol < args.poolW; ++ poolCol) {
				
				//	STUDENT: Calculate row and col of element here

				row = (row_gl * args.strideH) + poolRow;
				col = (col_gl * args.strideW) + poolCol;

				if(row >= 0 && row < inShape.height && 
						col >= 0 && col < inShape.width){

					float value = input[row * inShape.width + col];
					
					switch (args.opType)
					{
					//	STUDENT: Add cases and complete pooling code for all 3 options
					case PoolOp::MaxPool:

						if (value > poolPick)
						{	
							poolPick = value;
						}
						break;

					case PoolOp::MinPool:

						if (value < poolPick)
						{
							poolPick = value;
						}
						break;

					case PoolOp::AvgPool:

						poolPick += value;
						poolPick = poolPick/(args.poolH * args.poolW);
						break;

					default:
						return;	
						break;
					}
				}
			}
		}

		output[row_gl * outShape.width + col_gl] = poolPick;

	}
}


int runGpuPool (TensorShape inShape, PoolLayerArgs poolArgs){
	
	float *input_d, *output_d;
	
	float* inMatrix = (float*) malloc(inShape.height * inShape.width * sizeof(float));

	if (inMatrix == NULL){
		std::cout<< "ERROR ERROR!!!!! RUN FOR THE HILLS!!!!!";
		return 0;
	} 

	TensorShape_t outShape;

	outShape.height = (inShape.height - poolArgs.poolH) / poolArgs.strideH + 1;		
	outShape.width = (inShape.width - poolArgs.poolW) / poolArgs.strideW + 1;
	outShape.channels = inShape.channels;

	std::cout<< "Output Dimensions: " << outShape.height << " * " << outShape.width <<"\n";

	float* outMatrix =  (float*) malloc(outShape.height * outShape.width * sizeof(float));

	if (outMatrix == NULL){
		std::cout<< "ERROR ERROR!!!!! RUN FOR THE HILLS!!!!!";
		return -1;
	} 

	if(cudaMalloc(&input_d,  inShape.height * inShape.width * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< inShape.height * inShape.width * sizeof(float);
		std::cout<< "\n ERROR ERROR!!!!! RUN FOR THE HILLS!!!!! INPUT MEMORY ALLOCATION FAILURE \n";
		return -1;
	}
	
	if(cudaMalloc(&output_d, outShape.height * outShape.width * sizeof(float))!=cudaSuccess){
		std::cout<< "Size Requested: "<< outShape.height * outShape.width * sizeof(float);
		std::cout<< "ERROR ERROR!!!!! RUN FOR THE HILLS!!!!!OUTPUT MEMORY ALLOCATION FAILURE \n";
		return -1;
	}

	for (int i = 0; i < inShape.height; i++){
		for(int j = 0; j < inShape.width; j++){
			inMatrix[i * inShape.width + j] = (rand() / (RAND_MAX + 1.)) * 100;
			// std::cout << inMatrix[i * inShape.width + j] << " ";
		}
		// std::cout << "\n";
	}

	cudaMemcpy(input_d, inMatrix, inShape.height * inShape.width * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil((float)outShape.width / (float)dimBlock.x), ceil((float)outShape.height / (float)dimBlock.y));

	// STUDENT: call pool function
	poolLayer_gpu<<<dimGrid, dimBlock>>>(input_d, inShape, output_d, outShape, poolArgs);

	cudaMemcpy(outMatrix, output_d, outShape.height * outShape.width * sizeof(float), cudaMemcpyDeviceToHost);

	// for (int i = 0; i < outShape.height; i++){
	// 	for(int j = 0; j < outShape.width; j++){

	// 		//  std::cout << outMatrix[i*outShape.width + j] << " ";

	// 		// std::cout << outMatrix[i*outShape.width + j] << "@ (" << i << ", " << j << ")" << outShape.width << "\n";
	// 	}
	// 	// std::cout << "\n";
	// }
	
	free(inMatrix);
	free(outMatrix);

	cudaFree(output_d);
	cudaFree(input_d);

	return 0;
}

