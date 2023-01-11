#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "_OpenCLUtil.h"

static unsigned int dbg_counter = 0;
#define DEBUG_PRINT(dbg_message)                                                   \
	{                                                                              \
		printf("Debug Message: %s\t Debug Count: %d\n", dbg_message, dbg_counter); \
		dbg_counter++;                                                             \
	}

/* Matrix struct for program */
typedef struct FluidComputingMatrix
{
	// Matrix dimensions
	int *dim;
	// Pre-calculated size of matrix
	int total_size;
	// For how much time to run
	int iterations;
	// Current iteration matrix
	double *curr_matrix;
	// Next iteration matrix
	double *next_matrix;
	// Cell type matrix - fluid, non-fluid, etc
	char *type_matrix;

} FluidComputingMatrix;

FluidComputingMatrix *matrix;

/* OpenCL stuff*/
cl_context context;
cl_command_queue commandQueue;
cl_device_id deviceid;
cl_kernel kernel;

/* OpenCL memory */
cl_mem curr_matrix_cl;
cl_mem next_matrix_cl;
cl_mem type_matrix_cl;
cl_mem dim_cl;

/// @brief Updates the current matrix to the next matrix status
/// @param self
void update_matrix(FluidComputingMatrix *self)
{
	for (int i = 0; i < self->dim[0]; i++)
	{
		for (int j = 0; j < self->dim[1]; j++)
		{
			int temp_index = i * self->dim[1] + j;
			self->curr_matrix[temp_index] = self->next_matrix[temp_index];
		}
	}
}

/// @brief Allocates memory for FluidComputingMatrix pointer
// and the dimensions of the matrix
/// @return 1 if error, 0 if no error
int pre_allocate_matrix_memory()
{
	/* Allocating memory for the fluid computing matrix*/
	matrix = (FluidComputingMatrix *)malloc(sizeof(FluidComputingMatrix));
	if (matrix == NULL)
	{
		perror("Error allocating memory for 'FluidComputingMatrix'\n");
		return 1;
	}
	/* Allocating memory for the dimensions*/
	matrix->dim = (int *)malloc(sizeof(int) * 2);
	if (matrix->dim == NULL)
	{
		perror("Error allocating memory for 'matrix dimensions'\n");
		return 1;
	}
	return 0;
}

/// @brief Allocates memory for the rest of the FluidComputingMatrix,
// including current, next iteration and type of matrix
/// @return 1 if error, 0 if no error
int allocate_matrix_memory()
{
	matrix->total_size = matrix->dim[0] * matrix->dim[1];

	/* Allocating memory for the current iteration matrix*/
	matrix->curr_matrix = (double *)malloc(sizeof(double) * matrix->total_size);
	if (matrix->curr_matrix == NULL)
	{
		perror("Error allocating memory for 'curr_matrix'\n");
		return 1;
	}
	/* Allocating memory for the next iteration matrix*/
	matrix->next_matrix = (double *)malloc(sizeof(double) * matrix->total_size);
	if (matrix->next_matrix == NULL)
	{
		perror("Error allocating memory for 'next_matrix'\n");
		return 1;
	}
	/* Allocating memory for the cell type matrix*/
	matrix->type_matrix = (char *)malloc(sizeof(int) * matrix->total_size);
	if (matrix->type_matrix == NULL)
	{
		perror("Error allocating memory for 'type_matrix'\n");
		return 1;
	}
	return 0;
}

/// @brief Allocates memory in the device and moves the data inside it
/// @return 1 if error, 0 if no error
int allocate_device_memory()
{
	int rc;
	/* Allocate device memory */
	curr_matrix_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * matrix->total_size, NULL, &rc);
	handleError(rc, __LINE__, __FILE__);
	next_matrix_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * matrix->total_size, NULL, &rc);
	handleError(rc, __LINE__, __FILE__);
	type_matrix_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) * matrix->total_size, NULL, &rc);
	handleError(rc, __LINE__, __FILE__);
	dim_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 2, NULL, &rc);
	handleError(rc, __LINE__, __FILE__);
}

/// @brief Frees memory of the host
void cleanup()
{
	/* Frees up host memory */
	free(matrix->curr_matrix);
	free(matrix->next_matrix);
	free(matrix->type_matrix);
	free(matrix->dim);
	free(matrix);
}

/// @brief Frees memory of the device
void cleanup_device()
{
	/* Frees up device memory */
	clReleaseMemObject(curr_matrix_cl);
	clReleaseMemObject(next_matrix_cl);
	clReleaseMemObject(type_matrix_cl);
	clReleaseMemObject(dim_cl);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);
}

/// @brief Loads the data from a file formatted correctly inside the data structures
/// @param input_file_name name of the input file
/// @return 1 if error, 0 if no error
int load_matrix(char *input_file_name)
{
	FILE *file_fptr = fopen(input_file_name, "r");
	if (file_fptr == NULL)
	{
		perror("Error opening the input file!\n");
		return 1;
	}

	fscanf(file_fptr, "%d %d\n", &matrix->dim[0], &matrix->dim[1]);

	if (allocate_matrix_memory())
	{
		return 1;
	}

	for (int j = 0; j < matrix->dim[1]; j++)
	{
		for (int i = 0; i < matrix->dim[0]; i++)
		{
			int temp_index = i * matrix->dim[1] + j;
			fscanf(file_fptr, "%c %lf\n", &matrix->type_matrix[temp_index],
				   &matrix->curr_matrix[temp_index]);
		}
	}

	fscanf(file_fptr, "%d", &matrix->iterations);
	fclose(file_fptr);
	return 0;
}

/// @brief Stores the data to a file
/// @param output_file_name name of the output file
/// @return 1 if error, 0 if no error
int store_results(char *output_file_name)
{
	FILE *file_fptr = fopen(output_file_name, "w");

	if (file_fptr == NULL)
	{
		perror("Error opening the output file!\n");
		return 1;
	}

	fprintf(file_fptr, "%d %d\n", matrix->dim[0], matrix->dim[1]);

	for (int j = 0; j < matrix->dim[1]; j++)
	{
		for (int i = 0; i < matrix->dim[0]; i++)
		{
			int temp_index = i * matrix->dim[1] + j;
			fprintf(file_fptr, "%c %lf\n", matrix->type_matrix[temp_index],
					matrix->next_matrix[temp_index]);
		}
	}
	fclose(file_fptr);
	return 0;
}

/// @brief Loads the runtime arguments inside data structures
/// @param argc
/// @param argv
/// @param input_file_name name of the input file
/// @param output_file_name name of the output file
/// @param worker_count how many worker items/GPU threads to use
/// @param worker_group_size how many worker items are inside a group
/// @return 1 if error, 0 if no error
int get_args(int argc, char **argv, char *input_file_name, char *output_file_name,
			 size_t *worker_count, size_t *worker_group_size)
{
	if (argc != 5)
	{
		perror("Usage: ./homework input_file.txt output_file.txt worker_count worker_group_size\n");
		return 1;
	}

	input_file_name = strdup(argv[1]);
	output_file_name = strdup(argv[2]);
	*worker_count = atoi(argv[3]);
	*worker_group_size = atoi(argv[4]);

	return 0;
}

/// @brief Moves data from FluidComputingMatrix data structure inside the memory of the device, sets the arguments inside the device kernel and the maximum worker item count
/// @param worker_group_size
/// @return
int setup_iteration(size_t *worker_group_size)
{
	int rc;
	size_t max_work_group_size;

	// Move data from host to device
	rc = clEnqueueWriteBuffer(commandQueue, curr_matrix_cl, CL_TRUE, 0, sizeof(double) * matrix->total_size, matrix->curr_matrix, 0, NULL, NULL);
	handleError(rc, __LINE__, __FILE__);
	rc = clEnqueueWriteBuffer(commandQueue, next_matrix_cl, CL_TRUE, 0, sizeof(double) * matrix->total_size, matrix->next_matrix, 0, NULL, NULL);
	handleError(rc, __LINE__, __FILE__);
	rc = clEnqueueWriteBuffer(commandQueue, type_matrix_cl, CL_TRUE, 0, sizeof(char) * matrix->total_size, matrix->type_matrix, 0, NULL, NULL);
	handleError(rc, __LINE__, __FILE__);
	rc = clEnqueueWriteBuffer(commandQueue, dim_cl, CL_TRUE, 0, sizeof(int) * 2, matrix->dim, 0, NULL, NULL);
	handleError(rc, __LINE__, __FILE__);

	// Set the arguments to our compute kernel
	rc = clSetKernelArg(kernel, 0, sizeof(cl_mem), &curr_matrix_cl);
	handleError(rc, __LINE__, __FILE__);
	rc = clSetKernelArg(kernel, 1, sizeof(cl_mem), &type_matrix_cl);
	handleError(rc, __LINE__, __FILE__);
	rc = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dim_cl);
	handleError(rc, __LINE__, __FILE__);
	rc = clSetKernelArg(kernel, 3, sizeof(cl_mem), &next_matrix_cl);
	handleError(rc, __LINE__, __FILE__);

	// Get the maximum work group size for executing the kernel on the device
	rc = clGetKernelWorkGroupInfo(kernel, deviceid, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
	handleError(rc, __LINE__, __FILE__);

	*worker_group_size = fmin(*worker_group_size, max_work_group_size);

	return 0;
}

int main(int argc, char **argv)
{
	int rc;
	char *input_file_name, *output_file_name;
	size_t worker_count, worker_group_size;

	if (get_args(argc, argv, input_file_name, output_file_name,
				 &worker_count, &worker_group_size))
	{
		return -1;
	}
	if (pre_allocate_matrix_memory())
	{
		return -1;
	}

	if (load_matrix(argv[1]))
	{
		return -1;
	}

	deviceid = initOpenCL(&context, &commandQueue);
	kernel = getAndCompileKernel("homework.cl", "temperature_calculations", context, deviceid);

	allocate_device_memory();

	for (int epoch = 0; epoch < matrix->iterations; epoch++)
	{
		if (setup_iteration(&worker_group_size))
		{
			return -1;
		}

		// Execute kernel
		rc = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &worker_count, &worker_group_size, 0, NULL, NULL);
		handleError(rc, __LINE__, __FILE__);

		// Wait for the command commands to get serviced before reading back results
		clFinish(commandQueue);

		// Move data from device to host memory
		rc = clEnqueueReadBuffer(commandQueue, next_matrix_cl, CL_TRUE, 0, sizeof(double) * matrix->total_size, matrix->next_matrix, 0, NULL, NULL);
		handleError(rc, __LINE__, __FILE__);

		update_matrix(matrix);
	}

	if (store_results(argv[2]))
	{
		return -1;
	}

	cleanup_device();
	cleanup();
	return 0;
}