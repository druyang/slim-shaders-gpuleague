//////////////////////////////////////////////////////////////////////////
////This is the code implementation for Hanon finger exercise -- memory
////Dartmouth COSC89.25/189.03, GPU Programming and High-Performance Computing
//////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team="Team_X";
	std::string author_1="Name_1";
	std::string author_2="Name_2";
	std::string author_3="Name_3";	////optional
};

ofstream out;

//////////////////////////////////////////////////////////////////////////
////Hanon finger exercise for memory manipulations
////In this exercise you will practice the use of a set of CUDA memory APIs, 
////  including cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyFrom(To)Symbol, and cudaGetSymbolAddress

const int a_host[8]={1,2,3,4,5,6,7,8};								////a_host is an array on host
__device__ const int b_dev[8]={101,102,103,104,105,106,107,108};	////b_dev is an array on device

////Hanon Exercise 12: practice cudaMalloc, cudaMemcpy, and cudaFree
////Expected output: copy a_host from host to device, add each of its elements by 1, store the results in result_host
////Hint:
////0) allocate an array on device with the same size as a_host;
////1) copy a_host from host to device;
////2) write a kernel function to carry out the incremental operation on device;
////3) copy the calculated results on device to result_host (on host)
////4) free the array on device

/*TODO: Your kernel function starts*/
/*TODO: Your kernel function ends*/

__host__ void Hanon_Exercise_12()
{
	int result_host[8]={0};
	
	/*TODO: Your implementation starts*/
	/*TODO: Your implementation ends*/

	cout<<"Hanon exercise 12:\n";
	for(int i=0;i<8;i++)cout<<result_host[i]<<", ";cout<<endl;
	out<<"Hanon exercise 12:\n";
	for(int i=0;i<8;i++)out<<result_host[i]<<", ";out<<endl;
}

////Hanon Exercise 13: practice cudaMemcpyFromSymbol
////Expected output: copy b_dev to result_host by using cudaMemcpyFromSymbol. 
////Hint: b_dev is in static (stack) memory, so you cannot use cudaMemcpy to manipulate it!
__host__ void Hanon_Exercise_13()
{
	vector<int> result_host(8,0);
	
	/*TODO: Your implementation starts*/
	/*TODO: Your implementation ends*/

	cout<<"Hanon exercise 13:\n";
	for(int i=0;i<8;i++)cout<<result_host[i]<<", ";cout<<endl;
	out<<"Hanon exercise 13:\n";
	for(int i=0;i<8;i++)out<<result_host[i]<<", ";out<<endl;
}

////Hanon Exercise 14: practice manipulating dynamic and static memories together
////Expected output: calculate a_host+b_dev (element-wise sum) on device and store the results in result_host
////Hint:
////1) transferring a_host from host to device;
////2) write a kernel function to carry out the element-wise sum for arrays a_host and b_dev
////3) transfer the results from device to result_host (on host)

/*TODO: Your kernel function starts*/
/*TODO: Your kernel function ends*/

__host__ void Hanon_Exercise_14()
{
	int result_host[8]={0};
	
	/*TODO: Your host function implementation starts*/
	/*TODO: Your host function implementation ends*/

	cout<<"Hanon exercise 14:\n";
	for(int i=0;i<8;i++)cout<<result_host[i]<<", ";cout<<endl;
	out<<"Hanon exercise 14:\n";
	for(int i=0;i<8;i++)out<<result_host[i]<<", ";out<<endl;
}

////Hanon Exercise 15: practice using shared memory
////Expected output: calculate a_host*s+b_dev and store results in result_host. Here s is an array initialized in shared memory of the kernel function
////Hint: You need to modify the arguments and the implementation of the function Calculate_Array_With_Shared() to pass in your array(s) and perform calculations 

__global__ void Calculate_Array_With_Shared()	/*TODO: modify the arguments of the kernel function*/
{
	__shared__ int s[8];
	s[threadIdx.x]=2*threadIdx.x;
	__syncthreads();

	/*TODO: Your kernel implementation starts*/
	/*TODO: Your kernel implementation ends*/
}

__host__ void Hanon_Exercise_15()
{	
	/*TODO: Your host function implementation starts*/
	/*TODO: Your host function implementation ends*/

	int result_host[8]={0};	

	cout<<"Hanon exercise 15:\n";
	for(int i=0;i<8;i++)cout<<result_host[i]<<", ";cout<<endl;
	out<<"Hanon exercise 15:\n";
	for(int i=0;i<8;i++)out<<result_host[i]<<", ";out<<endl;
}

////Hanon Exercise 15: practice cudaGetSymbolAddress
////Expected output: apply the following kernel function Manipulate_Array() onto b_dev and store the results in result_host
////*WITHOUT* modifying the implementation in Manipulate_Array() (call it as a blackbox)
////Hint: get the dynamic address for b_dev by calling cudaGetSymbolAddress

////Note: You are not allowed to modify the implementation in this function!
__global__ void Manipulate_Array(int* array)
{
	array[threadIdx.x]*=16;
	array[threadIdx.x]+=1;
}

__host__ void Hanon_Exercise_16()
{
	/*TODO: Your host function implementation starts*/
	/*TODO: Your host function implementation ends*/

	int result_host[8]={0};	

	cout<<"Hanon exercise 16:\n";
	for(int i=0;i<8;i++)cout<<result_host[i]<<", ";cout<<endl;
	out<<"Hanon exercise 16:\n";
	for(int i=0;i<8;i++)out<<result_host[i]<<", ";out<<endl;
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
}

int main()
{
	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_exercise_thread.dat";
	out.open(file_name);
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	Hanon_Exercise_Test_Memory();
	return 0;
}