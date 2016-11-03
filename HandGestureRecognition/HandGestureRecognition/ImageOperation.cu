
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>

#include <iostream>
#include <time.h>
#include <Windows.h>
#include <vector>

#include "utilites.h"


#define BBOX_SIZE 32
#define BLOCK_SIZE_REDUCTION 128
#define BLOCK_SIZE 256
#define IMAGE_WIDTH 150
#define IMAGE_HEIGTH 150


void train_averages_dataset(unsigned int elem_size);
void test_averages_dataset(unsigned int elem_size);
void test_covariances_dataset(unsigned int elem_size);

std::vector<cv::Mat> read_image_dir(std::string dir);

std::vector<cv::Mat> test_differences(cv::Mat input, unsigned int elem_size);
void test_compute_pixel_average(unsigned int elem_size);
std::vector<unsigned int> test_compute_reduction_sum(std::vector<cv::Mat> differences, unsigned int elem_size);
void test_compute_pca(unsigned int elem_size);
void test_image_correlation();
void test_image_covariance();




std::vector<std::string> directories { "A", "B", "C", "D", "E", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y" };



__global__ void difference_kernel(unsigned char* input1, unsigned char* input2, unsigned char* output, unsigned int elem_size)
{
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < elem_size; i += gridDim.x*blockDim.x)
		output[i] = input1[i] > input2[i] ? input1[i] - input2[i] : input2[i] - input1[i];
}

__global__ void pixel_average_kernel(unsigned char* input, float * output, unsigned int n, unsigned int elem_size)
{
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < elem_size; i += gridDim.x*blockDim.x)
		output[i] += (float) input[i]/n;
}

__global__ void covariance_kernel(unsigned char * input0, unsigned char * input1, unsigned char input0_avg, unsigned char input1_avg, float * output, unsigned int elem_size){
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < elem_size; i += gridDim.x*blockDim.x)
		output[i] +=  (float)  ((input0[i] - input0_avg) * (input1[i]-input1_avg))   / (float) elem_size  ;
}

__global__ void binary_kernel(unsigned char* input, unsigned char* output, unsigned char threshold, unsigned int elem_size)
{
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < elem_size; i += gridDim.x*blockDim.x)
		output[i] = input[i] > threshold ? 255:0;
}

__global__ void substract(unsigned int * input, int * output, float substract_value, unsigned int elem_size){
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < elem_size; i += gridDim.x*blockDim.x)
		output[i] = input[i] - substract_value;
}

__global__ void add(unsigned int * input, int * output, float add_value, unsigned int elem_size){
	for (int i = threadIdx.x + blockDim.x*blockIdx.x; i < elem_size; i += gridDim.x*blockDim.x)
		output[i] = input[i] + add_value;
}

// It is not as efficient as it could be
__global__ void reduction_sum(unsigned char * input, unsigned int * output, int size) {
	__shared__ int sdata[2048];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	sdata[tid] = input[i];
	__syncthreads();
	// do reduction in shared mem
	for	(unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
		__syncthreads();
		if (blockIdx.x == 0)
			for (int k = 1; k < gridDim.x; k++)
				output[0] += output[k];
	}
}
__global__ void reduction_sum_float(float * input, float * output, int size) {

	__shared__ float sdata[2048];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	sdata[tid] = input[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
		__syncthreads();
		if (blockIdx.x == 0)
		for (int k = 1; k < gridDim.x; k++)
			output[0] += output[k];
	}
	
}


// Compute the average of a vector of images with the same size.
cudaError_t pixel_average(std::vector<cv::Mat> input_v, float * output_average, unsigned int elem_size){

	cudaError_t cudaStatus;

	float * dev_output_average = 0;
	unsigned char * dev_image = 0;

	//Allocate memory for the all the images GPU buffer (only one at a time will be loaded).
	cudaStatus = cudaMalloc((void**)&dev_image, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Allocate memory for the output image GPU buffer.
	cudaStatus = cudaMalloc((void**)&dev_output_average, elem_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemset(dev_output_average, 0, elem_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	unsigned int elem_count = input_v.size();
	for (cv::Mat image : input_v){
		//Copy the image to GPU buffer
		cudaStatus = cudaMemcpy(dev_image, image.data, elem_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		//Launch the kernel
		pixel_average_kernel << <256,256 >> >(dev_image, dev_output_average, elem_count, elem_size);

	}

	// Copy output average from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output_average, dev_output_average, elem_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_image);
	cudaFree(dev_output_average);

	return cudaStatus;
}

cudaError_t image_average(unsigned char * input_image, unsigned char * output, unsigned int elem_size){

	cudaError_t cudaStatus;

	unsigned char * dev_input_image = 0;
	unsigned int * dev_output = 0;
	unsigned int * host_output_average = 0;

	//Allocate memory for the input image in the GPU buffer.
	cudaStatus = cudaMalloc((void**)&dev_input_image, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Allocate memory for the output in the GPU buffer.
	cudaStatus = cudaMalloc((void**)&dev_output, elem_size * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_input_image, input_image, elem_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//sum_array_kernel << <128,128 >> >(dev_input_image, elem_size);
	reduction_sum << <BLOCK_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >> >(dev_input_image, dev_output, elem_size);

	//Copy the result from device to host
	host_output_average = (unsigned int *)malloc(elem_size * sizeof(unsigned int));
	cudaStatus = cudaMemcpy(host_output_average, dev_output, elem_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/*
	// Print image values
	for (int i = 0; i < elem_size; i++)
		std::cout << host_output_average[i] << ", ";
	std::cout << std::endl;
	*/

	*output = (unsigned char) (host_output_average[0] / elem_size);

Error:
	cudaFree(dev_input_image);
	cudaFree(dev_output);
	return cudaStatus;
}

cudaError_t image_covariance(unsigned char * input0, unsigned char * input1, unsigned char input0_avg, unsigned char input1_avg, float * output, unsigned int elem_size){

	cudaError_t cudaStatus;

	unsigned char * dev_input0_image = 0;
	unsigned char * dev_input1_image = 0;
	float * dev_covariance = 0;
	float * dev_output = 0;
	float * host_output_covariance = 0;

	//Allocate memory for the input images in the GPU buffer.
	cudaStatus = cudaMalloc((void**)&dev_input0_image, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_input1_image, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Allocate memory for the output in the GPU buffer.
	cudaStatus = cudaMalloc((void**)&dev_covariance, elem_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_output, elem_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_input0_image, input0, elem_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_input1_image, input1, elem_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	covariance_kernel << <BLOCK_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >> >(dev_input0_image, dev_input1_image, input0_avg, input1_avg, dev_covariance, elem_size);
	reduction_sum_float << <BLOCK_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >> >(dev_covariance, dev_output, elem_size);

	//Copy the result from device to host
	host_output_covariance = (float *)malloc(elem_size * sizeof(float));
	cudaStatus = cudaMemcpy(host_output_covariance, dev_output, elem_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/*
	// Print image values
	for (int i = 0; i < elem_size; i++)
		std::cout << host_output_covariance[i] << ", ";
	std::cout << std::endl;
	*/

	*output =  host_output_covariance[0];

Error:
	cudaFree(dev_input0_image);
	cudaFree(dev_input1_image);
	cudaFree(dev_covariance);
	cudaFree(dev_output);
	free(host_output_covariance);

	return cudaStatus;
}

cudaError_t image_correlation(unsigned char * input_img0, unsigned char * input_img1, float * output_correlation_img, unsigned int size0_rows, unsigned int size0_cols, unsigned int size1_rows, unsigned int size1_cols){
	cudaError_t cudaStatus;
	
	unsigned char average_img0;
	unsigned char average_img1;

	cudaStatus = image_average(input_img0, &average_img0, size0_rows * size0_cols);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "image_average failed!");
		goto Error;
	}

	cudaStatus = image_average(input_img1, &average_img1, size1_rows * size1_cols);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "image_average failed!");
		goto Error;
	}


	float std_dev_0 = 0.0f;
	for (int i = 0; i < size0_rows * size0_cols; i++){
		std_dev_0 += (float) ((input_img0[i] - average_img0) * (input_img0[i] - average_img0)) / (size0_rows * size0_cols);
	}

	float std_dev_1 = 0.0f;
	for (int i = 0; i < size1_rows * size1_cols; i++)
		std_dev_1 += (float)((input_img1[i] - average_img1) * (input_img1[i] - average_img1)) / (size1_rows * size1_cols);

	std_dev_0 = std::sqrt(std_dev_0);
	std_dev_1 = std::sqrt(std_dev_1);



	
	for (int offset_i = 0; offset_i <= size0_rows - size1_rows; offset_i++)
		for (int offset_j = 0; offset_j <= size0_cols - size1_cols; offset_j++){
			int offset = offset_i * size0_cols + offset_j;
			float cov = 0.0f;
			for (int i = 0; i < size1_rows; i++)
				for (int j = 0; j < size1_cols; j++){
					int idx = i * size1_cols + j;
				cov += (input_img0[idx+offset] - average_img0) * (input_img1[idx] - average_img1);
				}
			cov = cov / (size1_cols * size1_rows);
			output_correlation_img[offset] = cov / (std_dev_0*std_dev_1);
			//std::cout << output_correlation_img[offset] << ", ";
		}
	

Error:
	return cudaStatus;
}

std::vector<cv::Mat> differences_baricenters(cv::Mat input, unsigned int elem_size){
	unsigned char * dev_input = 0;
	unsigned char * dev_avg = 0;
	unsigned char * dev_output = 0;

	std::vector<cv::Mat> differences;
	std::vector<cv::Mat> averages = read_image_dir("Dataset/TrainedBaricenters");

	// Resize to match the standard images
	cv::Mat r_img;
	cv::resize(input, r_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGTH));


	//Allocate memory for the input image in the GPU buffer.
	cudaError cudaStatus = cudaMalloc(&dev_input, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_input, r_img.data, elem_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	//Allocate memory for the avg image in the GPU buffer.
	cudaStatus = cudaMalloc(&dev_avg, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Allocate memory for the input image in the GPU buffer.
	cudaStatus = cudaMalloc(&dev_output, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Compute the differences for each average image
	unsigned char * diff_data = (unsigned char *)malloc(elem_size * sizeof(unsigned char));
	for (cv::Mat avg : averages){


		cudaStatus = cudaMemcpy(dev_avg, avg.data, elem_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		// Launch the kernel
		difference_kernel << <BLOCK_SIZE, BLOCK_SIZE >> >(dev_input, dev_avg, dev_output, elem_size);

		cudaStatus = cudaMemcpy(diff_data, dev_output, elem_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		unsigned char * cpy = (unsigned char *)malloc(elem_size * sizeof(unsigned char));
		std::memcpy(cpy, diff_data, elem_size * sizeof(unsigned char));
		cv::Mat diff = cv::Mat(IMAGE_WIDTH, IMAGE_HEIGTH, CV_8U, cpy);
		differences.push_back(diff);
		diff.release();
	}

Error:
	cudaFree(dev_input);
	cudaFree(dev_avg);
	cudaFree(dev_output);
	free(diff_data);

	return differences;
}

std::vector<float> covariances_baricenters(cv::Mat input, unsigned int elem_size){
	
	std::vector<float> covariances;
	std::vector<cv::Mat> averages = read_image_dir("Dataset/TrainedBaricenters");

	// Resize to match the standard images
	cv::Mat r_img;
	cv::resize(input, r_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGTH));

	unsigned char input_avg;
	image_average(input.data, &input_avg, elem_size);

	//Compute the covariances for each average image
	for (cv::Mat avg : averages){

		
		unsigned char avg_avg = 0;
		
		image_average(avg.data, &avg_avg, elem_size);

		float covariance = 0.0f;
		image_covariance(input.data, avg.data, input_avg, avg_avg, &covariance, elem_size);

		covariances.push_back(covariance);
	}

	return covariances;
}

cv::Rect find_bounding_box(cv::Mat input){
	double t = 1;
	cv::Mat binary;

	cv::threshold(input, binary, t, 255, cv::THRESH_BINARY);
	cv::Mat points;
	cv::findNonZero(binary, points);
	return cv::boundingRect(points);
}

int main()
{
	cudaError_t cudaStatus;
	cv::Mat image1, image2, image3;
	std::vector<std::vector<cv::Mat>> images;
	std::string dir = "CroppedGestures/Y";
	//for (int i = 0; i < directories.size(); i++){
		//images.push_back(read_image_dir(dir));
		//std::cout << directories.at(i) << " folder done!" << std::endl;
	//}
	
	/*
	image1 = cv::imread("A1/A (1).png", CV_LOAD_IMAGE_GRAYSCALE);
	images.push_back((unsigned char*) image1.data);
	
	image2 = cv::imread("A1/A (2).png", CV_LOAD_IMAGE_GRAYSCALE);
	images.push_back((unsigned char*) image2.data);

	image3 = cv::imread("A1/A (3).png", CV_LOAD_IMAGE_GRAYSCALE);
	images.push_back((unsigned char*) image3.data);
	*/
	
	int cols = IMAGE_WIDTH; int rows = IMAGE_HEIGTH;

	
	// Train the system (Averages)
	//train_averages_dataset(cols*rows);

	// Test the system (Averages)
	//test_averages_dataset(cols*rows);

	// Test the system (Covariances)
	test_covariances_dataset(cols*rows);

	
	
	
	// Compute pixel average in parallel.
	//test_compute_pixel_average(cols*rows);

	
	
	// Compute image average in parallel.
	//unsigned char img_average;
	//cudaStatus = image_average(img.data, &img_average, rows*cols);
	

	// Compute image correlation
	//int * correlation = (int *)malloc((cols + cols)*(rows + rows) * sizeof(int));
	//image_correlation(img0.data, img1.data, correlation, (cols + cols)*(rows + rows), cols*rows);
	/*cv::Mat correlation = cv::Mat::zeros(cols + cols, rows + rows, CV_8U);
	cv::matchTemplate(img0, img1, correlation, CV_TM_CCORR_NORMED);
	double min, max;
	cv::minMaxLoc(correlation, &min, &max);
	std::cout << "Min: " << min << ", Max: " << max << std::endl;
	cv::threshold(correlation, correlation, 0.999, 1, cv::THRESH_BINARY);
	*/
	/*
	for (int i = 0; i < (cols + cols)*(rows + rows); i++)
		std::cout << correlation[i] << ", ";
	std::cout << std::endl
	;
	cv::imshow("Correlation image", correlation);
	cv::waitKey(30);
	*/

	
	//Test differences
	//cv::Mat input = cv::imread("GestureDatabase/Y/Y (2).png", CV_LOAD_IMAGE_GRAYSCALE);
	//std::vector<cv::Mat> differences = test_differences(input, cols*rows);
	
	
	// Test the sum of the differences
	//test_compute_reduction_sum(differences, cols*rows);

	// Test the PCA
	//test_compute_pca(cols*rows);

	//Test image correlation
	//test_image_correlation();

	// Test image covariance
	//test_image_covariance();

	//cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//std::cout << "Image average: " << (unsigned int) img_average << std::endl;
	
	/*
	cv::Mat output_image(cols, rows, CV_8U, average_image);
	cv::imshow("Average image", output_image);
	cv::waitKey(30);
	*/
	
	std::cin.get();
	
	return 0;
}



void train_averages_dataset(unsigned int elem_size){
	std::vector<cv::Mat> result_set;
	cudaError_t cudaStatus;

	std::string dir = "Dataset/Training/";
	std::vector<std::vector<cv::Mat>> training_set;
	// Read images from the directory
	for (int i = 0; i < directories.size(); i++){
		training_set.push_back(read_image_dir(dir + directories.at(i)));
	}

	// Compute the average value for each pixel
	float * average_image = (float *)malloc(elem_size * sizeof(float));
	std::vector<cv::Mat> averages;
	for (std::vector<cv::Mat> v : training_set){
		cudaStatus = pixel_average(v, average_image, elem_size);

		unsigned char * cpy = (unsigned char *)malloc(elem_size * sizeof(unsigned char));
		for (int i = 0; i < elem_size; i++){
			cpy[i] = (unsigned char)std::ceil(average_image[i]);
		}
		averages.push_back(cv::Mat(IMAGE_WIDTH, IMAGE_HEIGTH, CV_8U, cpy));
	}

	// Write the averag images
	for (int i = 0; i < directories.size(); i++)
		cv::imwrite("Dataset/TrainedBaricenters/" + directories.at(i) + ".png", averages.at(i));
}

void test_averages_dataset(unsigned int elem_size){
	std::cout << "Starting average error testing..." << std::endl;

	std::string dir = "Dataset/Testing";
	std::vector<cv::Mat> testing_set;
	std::vector<std::string> testing_set_classes;

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	// Load the testing images and get their classes
	hFind = FindFirstFile((dir + "/*.png").c_str(), &FindFileData);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do{
			std::string filename = dir + "/" + FindFileData.cFileName;
			std::string imageClass = getClassFromFilename(FindFileData.cFileName);
			cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

			testing_set.push_back(img);
			testing_set_classes.push_back(imageClass);
		} while (FindNextFile(hFind, &FindFileData));
	}
	else
	{
		std::cout << "Error reading directory files." << std::endl;
		FindClose(hFind);
	}
	FindClose(hFind);

	// Load the trained images
	//std::vector<cv::Mat> trained_set = read_image_dir("Dataset/TrainedBaricenters");


	/* TEST THE ACCURACY OF THE SYSTEM */
	int hits = 0;
	for (int i = 0; i < testing_set.size(); i++){
		cv::Mat input = testing_set.at(i);
		std::vector<cv::Mat> differences = differences_baricenters(input, elem_size);
		std::vector<float> average_diff_value;

		std::vector<unsigned int> difference_sum = test_compute_reduction_sum(differences, elem_size);
		for (int j = 0; j < difference_sum.size(); j++)
			average_diff_value.push_back(difference_sum.at(j) / elem_size);

		auto it_min = std::min_element(average_diff_value.begin(), average_diff_value.end());
		int class_index = it_min - average_diff_value.begin();
		if (testing_set_classes.at(i) == directories.at(class_index))
			hits++;
	}

	std::cout << "Hits: " << hits << std::endl;
	float precision = 100 * (float)hits / (float)testing_set.size();
	std::cout << "Precision: " << precision << "%" << std::endl;

	std::cout << "Average error testing finished!" << std::endl << std::endl;
}

void test_covariances_dataset(unsigned int elem_size){
	std::cout << "Starting covariance testing..." << std::endl;

	std::string dir = "Dataset/Testing";
	std::vector<cv::Mat> testing_set;
	std::vector<std::string> testing_set_classes;

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	// Load the testing images and get their classes
	hFind = FindFirstFile((dir + "/*.png").c_str(), &FindFileData);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do{
			std::string filename = dir + "/" + FindFileData.cFileName;
			std::string imageClass = getClassFromFilename(FindFileData.cFileName);
			cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

			testing_set.push_back(img);
			testing_set_classes.push_back(imageClass);
		} while (FindNextFile(hFind, &FindFileData));
	}
	else
	{
		std::cout << "Error reading directory files." << std::endl;
		FindClose(hFind);
	}
	FindClose(hFind);

	// Load the trained images
	//std::vector<cv::Mat> trained_set = read_image_dir("Dataset/TrainedBaricenters");


	/* TEST THE ACCURACY OF THE SYSTEM */
	int hits = 0;
	for (int i = 0; i < testing_set.size(); i++){
		cv::Mat input = testing_set.at(i);
		std::vector<float> covariances = covariances_baricenters(input, elem_size);

		auto it_min = std::max_element(covariances.begin(), covariances.end());
		int class_index = it_min - covariances.begin();
		if (testing_set_classes.at(i) == directories.at(class_index))
			hits++;
	}

	std::cout << "Hits: " << hits << std::endl;
	float precision = 100 * (float)hits / (float)testing_set.size();
	std::cout << "Precision: " << precision << "%" << std::endl;

	std::cout << "Covariance testing finished!" << std::endl << std::endl;
}



std::vector<cv::Mat> read_image_dir(std::string dir){
	std::vector<cv::String> filenames;
	std::vector<cv::Mat> image_data_v;

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	
	hFind = FindFirstFile((dir + "/*.png").c_str(), &FindFileData);
	if (hFind != INVALID_HANDLE_VALUE) 
	{
			do{
				std::string filename = dir + "/" + FindFileData.cFileName;
				std::string imageClass = getClassFromFilename(FindFileData.cFileName);
				cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
				
				/* For cropping and writing the images
				cv::Mat r_img;
				cv::resize(img, r_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGTH));
				cv::imwrite("CroppedGestures/" + dir + "/" + FindFileData.cFileName, r_img);
				*/

				image_data_v.push_back(img);
				//std::cout << FindFileData.cFileName << " loaded." << std::endl;
			} while (FindNextFile(hFind, &FindFileData));
	} 
	else 
	{
		std::cout << "Error reading directory files." << std::endl;
		FindClose(hFind);
		return image_data_v; 
	}

	FindClose(hFind);

	return image_data_v;
}



std::vector<cv::Mat> test_differences(cv::Mat input, unsigned int elem_size){
	unsigned char * dev_input = 0;
	unsigned char * dev_avg = 0;
	unsigned char * dev_output = 0;
	
	std::vector<cv::Mat> differences;
	std::vector<cv::Mat> averages = read_image_dir("Averages");

	// Resize to match the standard images
	cv::Mat r_img;
	cv::resize(input, r_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGTH));


	//Allocate memory for the input image in the GPU buffer.
	cudaError cudaStatus = cudaMalloc(&dev_input, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_input, r_img.data, elem_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	

	//Allocate memory for the avg image in the GPU buffer.
	cudaStatus = cudaMalloc(&dev_avg, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Allocate memory for the input image in the GPU buffer.
	cudaStatus = cudaMalloc(&dev_output, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Compute the differences for each average image
	unsigned char * diff_data = (unsigned char *)malloc(elem_size * sizeof(unsigned char));
	for (cv::Mat avg : averages){
		

		cudaStatus = cudaMemcpy(dev_avg, avg.data, elem_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		// Launch the kernel
		difference_kernel << <BLOCK_SIZE,BLOCK_SIZE >> >(dev_input, dev_avg, dev_output, elem_size);

		cudaStatus = cudaMemcpy(diff_data, dev_output, elem_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		unsigned char * cpy = (unsigned char *)malloc(elem_size * sizeof(unsigned char));
		std::memcpy(cpy, diff_data, elem_size * sizeof(unsigned char));
		cv::Mat diff = cv::Mat(IMAGE_WIDTH, IMAGE_HEIGTH, CV_8U, cpy);
		differences.push_back(diff);
		diff.release();
	}

	/*
	//Show the differences
	for (cv::Mat diff : differences){
		cv::imshow("Diff", diff);
		cv::waitKey(500);
	}
	*/

Error:
	cudaFree(dev_input);
	cudaFree(dev_avg);
	cudaFree(dev_output);
	free(diff_data);

	return differences;
}


void test_compute_pixel_average(unsigned int elem_size){
	std::vector<std::vector < cv::Mat > > images;
	cudaError_t cudaStatus;

	// Read images from the directory
	for (int i = 0; i < directories.size(); i++){
		images.push_back(read_image_dir("CroppedGestures/" + directories.at(i)));
	}

	// Compute the average value for each pixel
	float * average_image = (float *)malloc(elem_size * sizeof(float));
	std::vector<cv::Mat> averages;
	for (std::vector<cv::Mat> v : images){
		cudaStatus = pixel_average(v, average_image, elem_size);

		unsigned char * cpy = (unsigned char *)malloc(elem_size * sizeof(unsigned char));
		for (int i = 0; i < elem_size; i++){
			cpy[i] = (unsigned char) std::ceil(average_image[i]);
		}
		averages.push_back(cv::Mat(IMAGE_WIDTH, IMAGE_HEIGTH, CV_8U, cpy));
	}

	/*
	// Write the averag images
	for (int i = 0; i < directories.size(); i++)
		cv::imwrite("Averages/" + directories.at(i) + ".png", averages.at(i));
	*/

	for (cv::Mat avg : averages){
		cv::imshow("Avg", avg);
		cv::waitKey(100);
	}

Error:
	free(average_image);
}

std::vector<unsigned int> test_compute_reduction_sum(std::vector<cv::Mat> differences, unsigned int elem_size){
	unsigned char * dev_input = 0;
	unsigned int * dev_output = 0;

	std::vector<unsigned int> results;

	//Allocate memory for the input image in the GPU buffer.
	cudaError_t cudaStatus = cudaMalloc(&dev_input, elem_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Allocate memory for the output image in the GPU buffer.
	cudaStatus = cudaMalloc(&dev_output, elem_size * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	unsigned int * sum = (unsigned int*)malloc(elem_size * sizeof(unsigned int));
	for (int i = 0; i < differences.size(); i++){
		// Copy the data to the device
		cudaStatus = cudaMemcpy(dev_input, differences.at(i).data, elem_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		reduction_sum << <BLOCK_SIZE, BLOCK_SIZE >> >(dev_input, dev_output, elem_size);

		// Copy the data to the device
		cudaStatus = cudaMemcpy(sum, dev_output, elem_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!"); 
			goto Error;
		}
	
		results.push_back(sum[0]);
	}
	/*
	for (int i = 0; i < results.size(); i++){
		std::cout << "[" << i << "]: " << results.at(i) << std::endl;
	}
	*/

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);
	free(sum);

	return results;
}


void test_compute_pca(unsigned int elem_size){

	std::vector<std::vector<cv::Mat>> dir_imgs;
	int elem_count = 0;
	std::vector<cv::Mat> tmp;
	// Read images from the directory
	for (int i = 0; i < directories.size(); i++){
		tmp = read_image_dir("CroppedGestures/" + directories.at(i));
		elem_count += tmp.size();
		dir_imgs.push_back(tmp);
	}

	std::vector<cv::Mat> images;
	for (int i = 0; i < dir_imgs.size(); i++){
		tmp = dir_imgs.at(i);
		images.insert(images.end(), tmp.begin(), tmp.end());
	}
	dir_imgs.clear();

	cv::Mat * img_array = new cv::Mat[elem_count];
	cv::Mat data;
	
	for (int i = 0; i < elem_count; i++){
		img_array[i] = images.at(i).reshape(1,1);
	}
	images.clear();

	cv::vconcat(img_array, elem_count, data);

	// Compute the PCA
	int max_components = 100;
	cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, max_components);

	std::cout << "PCA finished!" << std::endl;

	cv::Mat compressed, reconstructed;
	compressed.create(data.cols, max_components, CV_32F);
	cv::Mat coeffs = compressed.row(0);

	pca.project(data.row(0), coeffs);
	pca.backProject(coeffs, reconstructed);

	std::cout << reconstructed.channels();

	reconstructed.rows = IMAGE_HEIGTH;
	reconstructed.cols = IMAGE_WIDTH;
	for (int i = 0; i < reconstructed.cols; i++){
		for (int j = 0; j < reconstructed.rows; j++){
			std::cout << reconstructed.at<float>(j, i)  << ",";
		}
		std::cout << std::endl;
	}
	cv::imshow("rec", reconstructed);
	cv::waitKey(2000);
}

void test_image_correlation(){
	cv::Mat img0 = cv::imread("squares.png", CV_LOAD_IMAGE_GRAYSCALE);// cv::Mat(100, 100, CV_8U, 2);
	cv::Mat img1 = cv::imread("square.png", CV_LOAD_IMAGE_GRAYSCALE);// cv::Mat(10, 10, CV_8U, 255);

	//unsigned int sizeZeros = IMAGE_WIDTH * IMAGE_HEIGTH;
	//unsigned int sizeOnes = IMAGE_WIDTH * IMAGE_HEIGTH;
	unsigned int sizeZeros = img0.cols * img0.rows;
	unsigned int sizeOnes = img1.cols * img1.rows;

	float * correlation_f_data = (float *)malloc(sizeZeros * sizeof(float));
	unsigned char * correlation_data = (unsigned char*)malloc(sizeZeros * sizeof(unsigned char));

	cudaError_t cudastatus;
	cudastatus = image_correlation(img0.data, img1.data, correlation_f_data, img0.rows, img0.cols, img1.rows, img1.cols);

	for (int i = 0; i < sizeZeros; i++){
		correlation_data[i] = (unsigned char) std::ceil(255 * correlation_f_data[i]);
	}

	std::cout << (int) correlation_data[0];


	cv::Mat correlation = cv::Mat(img0.rows, img0.cols, CV_8U, correlation_data);
	cv::transpose(correlation, correlation);
	cv::imshow("Correlation", correlation);
	cv::waitKey(5000);
}


void test_image_covariance(){
	cv::Mat img0 = cv::imread("Dataset/TrainedBaricenters/A.png", CV_LOAD_IMAGE_GRAYSCALE);// cv::Mat(100, 100, CV_8U, 2);
	cv::Mat img1 = cv::imread("Dataset/Testing/A (4).png", CV_LOAD_IMAGE_GRAYSCALE);// cv::Mat(10, 10, CV_8U, 255);

	unsigned int elem_size = img0.cols * img0.rows;

	float covariance;

	unsigned char img0_avg;
	unsigned char img1_avg;

	image_average(img0.data, &img0_avg, elem_size);
	image_average(img1.data, &img1_avg, elem_size);

	image_covariance(img0.data, img1.data, img0_avg, img1_avg, &covariance, elem_size);

	std::cout << "Covariance: " << covariance << std::endl << std::endl;
}