#include <cstdio>

__device__ void cuda_device_function()
{
	printf("This function is called from device only. a=%d, \n", blockIdx.x);
}


// I'm using this function to test array indices/outputs (total variable)
__global__ void cuda_global_function()
{
    int total = ((blockIdx.x+1)*(blockIdx.y+1))*(threadIdx.x+1)*(threadIdx.y+1); // sample/dummy variable 
	printf("Block X: %d Block Y: %d Block Z: %d Thread X: %d Thread Y: %d Thread Z: %d Total: %d \n",blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, total);
}

__host__ void host_function()
{
	printf("This function is called from host only.\n");
}

int main() {
	host_function();

    //cuda_global_function<<<1,4>>>(); 

	dim3 block_size=dim3(2,2,1);    // these are the dim3 variable parameters initialized in hanon_exercise_threads.cu 
	dim3 thread_size=dim3(4,4,1);
    cuda_global_function<<<block_size,thread_size>>>(); 
	
	return 0;
}
