//////////////////////////////////////////////////////////////////////////
////This is the code implementation for Hanon finger exercise -- memory
////Dartmouth COSC89.25/189.03, GPU Programming and High-Performance Computing
//////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

namespace name
{
std::string team = "Slim_Shaders";
std::string author_1 = "Andrw_Yang";
std::string author_2 = "Matthew_Kenney";
};

ofstream out;

//////////////////////////////////////////////////////////////////////////
////Hanon finger exercise for memory manipulations
////In this exercise you will practice the use of a set of CUDA memory APIs,
////  including cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyFrom(To)Symbol, and cudaGetSymbolAddress

const int a_host[8] = {1, 2, 3, 4, 5, 6, 7, 8};								////a_host is an array on host
__device__ const int b_dev[8] = {101, 102, 103, 104, 105, 106, 107, 108};	////b_dev is an array on device

////Hanon Exercise 12: practice cudaMalloc, cudaMemcpy, and cudaFree
////Expected output: copy a_host from host to device, add each of its elements by 1, store the results in result_host
////Hint:
////0) allocate an array on device with the same size as a_host;
////1) copy a_host from host to device;
////2) write a kernel function to carry out the incremental operation on device;
////3) copy the calculated results on device to result_host (on host)
////4) free the array on device

/*TODO: Your kernel function starts*/
__global__ void Hanon_kernel(int* to_increment)
{
    int array_id = blockDim.x * blockIdx.x + threadIdx.x;
    to_increment[array_id] = to_increment[array_id] + 1;
}
/*TODO: Your kernel function ends*/

__host__ void Hanon_Exercise_12()
{

    int result_host[8] = {0};
    int *a_dev = 0;

    /*TODO: Your implementation starts*/
    cudaMalloc((void**)&a_dev, 8 * sizeof(int));
    cudaMemcpy(a_dev, a_host, 8 * sizeof(int), cudaMemcpyHostToDevice);
    Hanon_kernel <<< 1, 8, 8 * sizeof(int)>>>(a_dev);
    cudaMemcpy(result_host, a_dev, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(a_dev);
    /*TODO: Your implementation ends*/

    cout << "Hanon exercise 12:\n";
    for(int i = 0; i < 8; i++)cout << result_host[i] << ", ";
    cout << endl;
    out << "Hanon exercise 12:\n";
    for(int i = 0; i < 8; i++)out << result_host[i] << ", ";
    out << endl;
}


////Hanon Exercise 13: practice cudaMemcpyFromSymbol
////Expected output: result_host={101,102,103,104,105,106,107,108}
////Process: copy b_dev (the static CUDA device array declared in line 35) to result_host by using cudaMemcpyFromSymbol.
////Hint: b_dev is in static (stack) memory, so you cannot use cudaMemcpy to manipulate it!
__host__ void Hanon_Exercise_13()
{
    vector<int> result_host(8, 0);

    /*TODO: Your implementation starts*/
    cudaMemcpyFromSymbol((void*)&result_host[0], b_dev, 8 * sizeof(int));
    /*TODO: Your implementation ends*/

    cout << "Hanon exercise 13:\n";
    for(int i = 0; i < 8; i++)cout << result_host[i] << ", ";
    cout << endl;
    out << "Hanon exercise 13:\n";
    for(int i = 0; i < 8; i++)out << result_host[i] << ", ";
    out << endl;
}

////Hanon Exercise 14: practice manipulating dynamic and static memories together
////Expected output: result_host={101+1,102+2,103+3,104+4,105+5,106+6,107+7,108+8}
////Process: calculate a_host+b_dev (element-wise sum) on device and store the results in result_host
////Hint:
////1) transferring a_host from host to device;
////2) write a kernel function to carry out the element-wise sum for arrays a_host and b_dev
////3) transfer the results from device to result_host (on host)

/*TODO: Your kernel function starts*/
__global__ void Hanon_kernel_14(int* to_increment)
{
    int array_id = blockDim.x * blockIdx.x + threadIdx.x;
    to_increment[array_id] = to_increment[array_id] + b_dev[array_id];
}
/*TODO: Your kernel function ends*/

__host__ void Hanon_Exercise_14()
{
    int result_host[8] = {0};

    int *a_dev = 0;
    /*TODO: Your implementation starts*/
    cudaMalloc((void**)&a_dev, 8 * sizeof(int));
    cudaMemcpy(a_dev, a_host, 8 * sizeof(int), cudaMemcpyHostToDevice);
    Hanon_kernel_14 <<< 1, 8, 8 * sizeof(int)>>>(a_dev);
    cudaMemcpy(result_host, a_dev, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(a_dev);
    /*TODO: Your host function implementation ends*/

    cout << "Hanon exercise 14:\n";
    for(int i = 0; i < 8; i++)cout << result_host[i] << ", ";
    cout << endl;
    out << "Hanon exercise 14:\n";
    for(int i = 0; i < 8; i++)out << result_host[i] << ", ";
    out << endl;
}

////Hanon Exercise 15: practice using shared memory
////Expected output: result_host={1*0+101,2*2+102,3*4+103,4*6+104,5*8+105,6*10+106,7*12+107,8*14+108}
////Process: calculate a_host*s+b_dev and store results in result_host. Here s is an array initialized in shared memory of the kernel function (line 111-113)
////Hint: You need to modify the arguments and the implementation of the function Calculate_Array_With_Shared() to pass in your array(s) and perform calculations

__global__ void Calculate_Array_With_Shared(int* array_from_host)	/*TODO: modify the arguments of the kernel function*/
{
    __shared__ int s[8];
    s[threadIdx.x] = 2 * threadIdx.x;
    __syncthreads();

    /*TODO: Your kernel implementation starts*/
    s[threadIdx.x] = s[threadIdx.x] * array_from_host[threadIdx.x] + b_dev[threadIdx.x];
    __syncthreads();

    array_from_host[threadIdx.x] = s[threadIdx.x];


    /*TODO: Your kernel implementation ends*/
}

__host__ void Hanon_Exercise_15()
{
    /*TODO: Your host function implementation starts*/
    int result_host[8] = {0};
    int *a_dev = 0;
    cudaMalloc((void**)&a_dev, 8 * sizeof(int));
    cudaMemcpy(a_dev, a_host, 8 * sizeof(int), cudaMemcpyHostToDevice);

    Calculate_Array_With_Shared <<< 1, 8, 8 * sizeof(int)>>>(a_dev);
    cudaMemcpy(result_host, a_dev, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(a_dev);
    /*TODO: Your host function implementation ends*/


    cout << "Hanon exercise 15:\n";
    for(int i = 0; i < 8; i++)cout << result_host[i] << ", ";
    cout << endl;
    out << "Hanon exercise 15:\n";
    for(int i = 0; i < 8; i++)out << result_host[i] << ", ";
    out << endl;
}

////Hanon Exercise 16: practice cudaGetSymbolAddress
////Expected output: result_host={101*16+1,102*16+1,103*16+1,...,108*16+1}
////Process: apply the following kernel function Manipulate_Array() onto b_dev and store the results in result_host
////*WITHOUT* modifying the implementation in Manipulate_Array() (call it as a blackbox)
////Hint: b_dev is a static array on GPU, you need to get its dynamic pointer by calling cudaGetSymbolAddress, and then send this pointer into the kernel function to update its values

////Note: You are not allowed to modify the implementation in this function!
__global__ void Manipulate_Array(int* array)
{
    array[threadIdx.x] *= 16;
    array[threadIdx.x] += 1;
}

__host__ void Hanon_Exercise_16()
{
    int result_host[8] = {0};

    /*TODO: Your host function implementation starts*/
    void* devPtr = 0;
    cudaGetSymbolAddress(&devPtr, b_dev); // get address of b_dev constant
    Manipulate_Array <<< 1, 8, 8 * sizeof(int)>>>((int*)devPtr); // Call Kernel function
    cudaMemcpyFromSymbol(result_host, b_dev, sizeof(int) * 8); // Load the result into result_host
    /*TODO: Your host function implementation ends*/


    cout << "Hanon exercise 16:\n";
    for(int i = 0; i < 8; i++)cout << result_host[i] << ", ";
    cout << endl;
    out << "Hanon exercise 16:\n";
    for(int i = 0; i < 8; i++)out << result_host[i] << ", ";
    out << endl;
}

////Hanon Exercise 17: practice using shared memory with multiple array types
////Expected output: array_int={208,206,204,202}, array_float={8.,6.,4.,2.},
//// i.e., reverse the order of the int array, multiply each element by 2, and copy its values to the float array (by type conversion),
//// and reverse the order of the float array, multiply each element by 2, and copy its values to the int array (by type conversion)
//// You need to implement this process by using a piece of shared memory holding both two arrays
////Hint: read the sample code we went through in class on Thursday, and mimic its steps as
////1. Initialize two array pointers with the types of int and float to different addresses of the shared memory
////2. Copy the values from array_int and array_float to the proper elements in shared memory
////3. synchronize threads
////4. Copy the values with the proper order and rescaling factor from each array in shared memory to global memory (array_int and array_float)

__global__ void Reverse_And_Multiply_Two_Arrays_With_Extern_Shared(int* array_int, const size_t array_int_size, float* array_float, const size_t array_float_size)
{
    extern __shared__ int shared_mem[];
    int* ai = (int*)&shared_mem[0];
    float* af = (float*)&shared_mem[array_int_size];

    /*Your implementation*/
    __syncthreads();

    // swap the types as we pass into shared memory
    ai[threadIdx.x] = (int) array_float[array_float_size - 1 - threadIdx.x] * 2;
    af[threadIdx.x] = (float) array_int[array_int_size - 1 - threadIdx.x] * 2.0;
    __syncthreads();

    // Copy manipulated values back to global memory
    array_int[threadIdx.x] = ai[threadIdx.x];
    array_float[threadIdx.x] = af[threadIdx.x];

}

__host__ void Hanon_Exercise_17()
{
    int array_int_host[4] = {1, 2, 3, 4};
    float array_float_host[4] = {101., 102., 103., 104.};

    int* array_int_dev = 0;
    float* array_float_dev = 0;
    cudaMalloc((void**)&array_int_dev, 4 * sizeof(int));
    cudaMalloc((void**)&array_float_dev, 4 * sizeof(float));
    cudaMemcpy(array_int_dev, array_int_host, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(array_float_dev, array_float_host, 4 * sizeof(float), cudaMemcpyHostToDevice);

    /*Your implementation: comment back the following code with the correct specification for shared memory size (by replacing the * with a proper number) */
    Reverse_And_Multiply_Two_Arrays_With_Extern_Shared <<< 1, 4, 4 * sizeof(int) +4 * sizeof(float) >>> (array_int_dev, 4, array_float_dev, 4);
	
	// Copy results back to host memory
    cudaMemcpy(array_int_host, array_int_dev, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(array_float_host, array_float_dev, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Hanon exercise 17:\n";
    for(int i = 0; i < 4; i++)cout << array_int_host[i] << ", ";
    cout << endl;
    for(int i = 0; i < 4; i++)cout << array_float_host[i] << ", ";
    cout << endl;

    out << "Hanon exercise 17:\n";
    for(int i = 0; i < 4; i++)out << array_int_host[i] << ", ";
    out << endl;
    for(int i = 0; i < 4; i++)out << array_float_host[i] << ", ";
    out << endl;
}

////Congratulations! You have finished all your Hanon exercises today!
//////////////////////////////////////////////////////////////////////////

void Hanon_Exercise_Test_Memory()
{
    Hanon_Exercise_12();
    Hanon_Exercise_13();
    Hanon_Exercise_14();
    Hanon_Exercise_15();
    Hanon_Exercise_16();
    Hanon_Exercise_17();
}

int main()
{
    if(name::team == "Team_X")
        {
            printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
            return 0;
        }

    std::string file_name = name::team + "_exercise_memory.dat";
    out.open(file_name);

    if(out.fail())
        {
            printf("\ncannot open file %s to record results\n", file_name.c_str());
            return 0;
        }

    Hanon_Exercise_Test_Memory();
    return 0;
}
