#include "cpuLib.h"


void dbprintf(const char* fmt...) {
	#ifndef DEBUG_PRINT_DISABLE
		va_list args;

		va_start(args, fmt);
		int result = vprintf(fmt, args);
		// printf(fmt, ...);
		va_end(args);
	#endif
	return;
}

void vectorInit(float* v, int size) {
	for (int idx = 0; idx < size; ++idx) {
		v[idx] = (float)(rand() % 100);
	}
}

int verifyVector(float* a, float* b, float* c, float scale, int size) {
	int errorCount = 0;
	for (int idx = 0; idx < size; ++idx) {
		if (c[idx] != scale * a[idx] + b[idx]) {
			++errorCount;
			#ifndef DEBUG_PRINT_DISABLE
				std::cout << "Idx " << idx << " expected " << scale * a[idx] + b[idx] 
					<< " found " << c[idx] << " = " << a[idx] << " + " << b[idx] << "\n";
			#endif
		}
	}
	return errorCount;
}

void printVector(float* v, int size) {
	int MAX_PRINT_ELEMS = 5;
	std::cout << "Printing Vector : \n"; 
	for (int idx = 0; idx < std::min(size, MAX_PRINT_ELEMS); ++idx){
		std::cout << "v[" << idx << "] : " << v[idx] << "\n";
	}
	std::cout << "\n";
}

/**
 * @brief CPU code for SAXPY accumulation Y += A * X
 * 
 * @param x 	vector x
 * @param y 	vector y - will get overwritten with accumulated results
 * @param scale scale factor (A)
 * @param size 
 */
void saxpy_cpu(float* x, float* y, float scale, uint64_t size) {
	for (uint64_t idx = 0; idx < size; ++idx) {
		y[idx] = scale * x[idx] + y[idx];
	}
}

int runCpuSaxpy(uint64_t vectorSize) {
	uint64_t vectorBytes = vectorSize * sizeof(float);

	printf("Hello Saxpy!\n");

	float * a, * b, * c;

	a = (float *) malloc(vectorSize * sizeof(float));
	b = (float *) malloc(vectorSize * sizeof(float));
	c = (float *) malloc(vectorSize * sizeof(float));

	if (a == NULL || b == NULL || c == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);
	//	C = B
	std::memcpy(c, b, vectorSize * sizeof(float));
	float scale = 2.0f;

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif

	//	C = A + B
	saxpy_cpu(a, c, scale, vectorSize);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	return 0;
}

/**
 * @brief CPU-based Monte-Carlo estimation of value of pi
 * 
 * @param iterationCount 	number of iterations of MC evaluation
 * @param sampleSize 		number of random points evaluated in each iteration
 * @return int 
 */
int runCpuMCPi(uint64_t iterationCount, uint64_t sampleSize) {

	std::random_device random_device;
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	float x, y;
	uint64_t hitCount = 0;
	uint64_t totalHitCount = 0;
	std::string str;

	auto tStart = std::chrono::high_resolution_clock::now();

	#ifndef DEBUG_PRINT_DISABLE
		std::cout << "Iteration: ";
	#endif

	for (int iter = 0; iter < iterationCount; ++ iter) {
		hitCount = 0;

		#ifndef DEBUG_PRINT_DISABLE
			str = std::to_string(iter);
			std::cout << str << std::flush;
		#endif

		//	Main CPU Monte-Carlo Code
		for (uint64_t idx = 0; idx < sampleSize; ++idx) {
			x = dist(random_device);
			y = dist(random_device);
			
			if ( int(x * x + y * y) == 0 ) {
				++ hitCount;
			}
		}

		#ifndef DEBUG_PRINT_DISABLE
			std::cout << std::string(str.length(),'\b') << std::flush;
		#endif
		totalHitCount += hitCount;
	}
	#ifndef DEBUG_PRINT_DISABLE
		std::cout << str << std::flush << "\n\n";
	#endif

	//	Calculate Pi
	float approxPi = ((double)totalHitCount / sampleSize) / iterationCount;
	approxPi = approxPi * 4.0f;
		
	std::cout << std::setprecision(10);
	std::cout << "Estimated Pi = " << approxPi << "\n";

		
	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}


std::ostream& operator<< (std::ostream &o,ImageDim imgDim) {
	return (
		o << "Image : " << imgDim.height  << " x " << imgDim.channels << " x "
			<< imgDim.channels << " x " << imgDim.pixelSize << " " 
	);
}

int loadBytesImage(std::string bytesFilePath, ImageDim &imgDim, uint8_t ** imgData ) {
	#ifndef DEBUG_PRINT_DISABLE
		std::cout << "Opening File @ \'" << bytesFilePath << "\' \n";
	#endif

	std::ifstream bytesFile;

	bytesFile.open(bytesFilePath.c_str(), std::ios::in | std::ios::binary);

	if (! bytesFile.is_open()) {
		std::cout << "Unable to open \'" << bytesFilePath << "\' \n";
		return -1;
	}

	ImageDim_t fileDim;
	bytesFile.read((char *) &fileDim, sizeof(fileDim));

	std::cout << "Found " << fileDim.height << " x " << fileDim.width
		<< " x " << fileDim.channels << " x " << fileDim.pixelSize << " \n";
	
	uint64_t numBytes = fileDim.height * fileDim.width * fileDim.channels;
	*imgData = (uint8_t *) malloc(numBytes * sizeof(uint8_t));
	if (imgData == nullptr) {
		std::cout << "Unable to allocate memory for image data \n";
		return -2;
	}

	bytesFile.read((char *) *imgData, numBytes * sizeof(uint8_t));

	std::cout << "Read " << bytesFile.gcount() << " bytes \n" ;

	imgDim.height		= fileDim.height;
	imgDim.width		= fileDim.width;
	imgDim.channels		= fileDim.channels;
	imgDim.pixelSize	= fileDim.pixelSize;

	bytesFile.close();
	
	return bytesFile.gcount();

}

int writeBytesImage (std::string outPath, ImageDim &imgDim, uint8_t * outData) {
	std::ofstream bytesFile;

	bytesFile.open(outPath.c_str(), std::ios::out | std::ios::binary);

	if (! bytesFile.is_open()) {
		std::cout << "Unable to open \'" << outPath << "\' \n";
		return -1;
	}

	uint64_t numBytes = imgDim.height * imgDim.width * imgDim.channels;
	bytesFile.write((char*) &imgDim, sizeof(imgDim));
	bytesFile.write((char *) outData, numBytes * sizeof(uint8_t));

	bytesFile.close();

}

int medianFilter_cpu (uint8_t * inPixels, ImageDim imgDim, uint8_t * outPixels, MedianFilterArgs args) {

	uint32_t startRow = (args.filterH - 1) / 2;
	uint32_t endRow = imgDim.height - ((args.filterH - 1) / 2);
	uint32_t startCol = (args.filterW - 1) / 2;
	uint32_t endCol = imgDim.width - ((args.filterW - 1) / 2);
	uint32_t inRow, inCol;

	std::vector <uint8_t> window;
	window.resize(args.filterH * args.filterW);

	for (uint32_t channel = 0; channel < imgDim.channels; ++ channel) {
		for (uint32_t outRow = startRow; outRow < endRow; ++ outRow) {
			for (uint32_t outCol = startCol; outCol < endCol; ++ outCol) {
				for (uint32_t filRow = 0; filRow < args.filterH; ++ filRow) {
					for (uint32_t filCol = 0; filCol < args.filterW; ++ filCol) {
						inRow = outRow - (args.filterH - 1) / 2 + filRow;
						inCol = outCol - (args.filterW - 1) / 2 + filCol;

						if(inRow < imgDim.height && inCol < imgDim.width){
							window[filRow * args.filterW + filCol] = inPixels[(inRow * imgDim.width + inCol) * imgDim.channels + channel];
						}	
					}
				}

				std::sort(window.begin(), window.end());

				*(outPixels + (outRow * imgDim.width + outCol) * imgDim.channels + channel) = 
					window[(args.filterH * args.filterW) / 2];
			}
		}
		std::cout << "Channel " << channel << " \n";
	}
}

int runCpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args) {
	ImageDim imgDim;
	uint8_t * imgData;
	
	int bytesRead = loadBytesImage(imgPath, imgDim, &imgData);
	int imgSize = imgDim.height * imgDim.width * imgDim.channels * imgDim.pixelSize;

	std::cout << "Size = " << imgSize << "\n";
	uint8_t * outData = (uint8_t *) malloc(imgSize * sizeof(uint8_t));

	auto tStart = std::chrono::high_resolution_clock::now();

	medianFilter_cpu(imgData, imgDim, outData, args);

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	writeBytesImage(outPath, imgDim, outData);
	return 0;
}


std::ostream& operator<< (std::ostream &o,PoolOp op) { 
	switch(op) {
	case PoolOp::MaxPool : return o << "MaxPool";
	case PoolOp::AvgPool : return o << "AvgPool";
	case PoolOp::MinPool : return o << "MinPool";
	default: return o<<"(invalid pool op)";
	}
}

int poolLayer_cpu (float * input, TensorShape inShape, 
				float * output, TensorShape outShape, PoolLayerArgs args) {
	
	uint32_t poolH = args.poolH;
	uint32_t poolW = args.poolW;

	uint32_t strideH = args.strideH;
	uint32_t strideW = args.strideW;

	uint32_t outputH = outShape.height;
	uint32_t outputW = outShape.width;

	uint32_t row, col;

	std::cout << args.opType << " : " << inShape.height << " x " << inShape.width 
		<< " with a " << poolH << " x " << poolW << " window -> " 
		<< outputH << " x " << outputW << "\n";

	for (uint32_t outRow = 0; outRow < outputH; ++ outRow) {
		for (uint32_t outCol = 0; outCol < outputW; ++ outCol) {

			//	STUDENT: Assign to first value of pool area
			float poolPick = input[outRow * strideH * inShape.width + outCol * strideW];

			for (uint32_t poolRow = 0; poolRow < poolH; ++ poolRow) {
				for (uint32_t poolCol = 0; poolCol < poolW; ++ poolCol) {
					
					//	STUDENT: Calculate row and col of element here

					row = (outRow * strideH) + poolRow;
					col = (outCol * strideW) + poolCol;

					float value = input[row * inShape.width + col];

					// std::cout << "Considered : " << value << " @ (" << row << ", "<< col << ")" << std::endl;
					
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
						poolPick = poolPick/(poolH * poolW);
						break;

					default:
						std::cout << "Pick max from pool, you must!\n";
						return 0;
						break;
					}
				}
			}

			output[outRow * outputW + outCol] = poolPick;

			std::cout << poolPick << " @ (" << outRow << ", " << outCol << ")\n";

			std::cout << "\n";

		}
	}

}

int runCpuPool (TensorShape inShape, PoolLayerArgs poolArgs) {
	
	srand(time(NULL));

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
		return 0;
	} 

	std::cout << "Actual Matrix \n";

	for (int i = 0; i < inShape.height; i++){
		for(int j = 0; j < inShape.width; j++){
			inMatrix[i * inShape.width + j] = (rand() / (RAND_MAX + 1.)) * 100;
			std::cout << inMatrix[i * inShape.width + j] << " ";
		}
		std::cout << "\n";
	}

	// STUDENT: call pool function
	poolLayer_cpu(inMatrix, inShape, outMatrix, outShape, poolArgs);

	for (int i = 0; i < outShape.height; i++){
		for(int j = 0; j < outShape.width; j++){

			std::cout << outMatrix[i*outShape.width + j] << "@ (" << i << ", " << j << ")" << outShape.width << "\n";

		}
		// std::cout << "\n";
	}

	free(inMatrix);
	free(outMatrix);

	return 0;
}


