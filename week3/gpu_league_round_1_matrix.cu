//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 1
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team="slim_shaders";
	std::string author_1="Andrw_Yang";
	std::string author_2="Matthew_Kenney";
};

////This is a matrix class to carry out linear algebra operations on both GPU and CPU
////It is the same as the sample code I showed in class on Week 3. 

////NOTICE: You do not have to change the implementation in this class. 
////But if you do want to change part of it for performance reasons, please let us known by writting a submission note on Canvas.

class Matrix{
public:
    int m=0;							////number of rows
    int n=0;							////number of columns
	vector<float> elements_on_host;		////we use a std::vector for the element array on host
    float* elements_on_dev=0;			////we use a pointer for the element array on device
	bool on_host=true;

	////constructors
	__host__ Matrix(){}

	__host__ Matrix(const int _m,const int _n,bool _on_host=true)
	{
		on_host=_on_host;
		if(on_host)Resize_On_Host(_m,_n);
		else Resize_On_Device(_m,_n);
	}

	////destructor
	__host__ ~Matrix()
	{
		if(!on_host&&elements_on_dev!=0) cudaFree(elements_on_dev);		
	}

	////Resize on host or device
	__host__ void Resize_On_Host(const int _m,const int _n)
	{
		if(m==_m&&n==_n)return;
		m=_m;
		n=_n;
		elements_on_host.resize(m*n);
	}

	__host__ void Resize_On_Device(const int _m,const int _n)
	{
		if(m==_m&&n==_n)return;
		m=_m;
		n=_n;
		if(elements_on_dev!=0)cudaFree(elements_on_dev);
		cudaMalloc((void**)&elements_on_dev,m*n*sizeof(float));
	}

	////random access a matrix element
	inline __host__ float& operator() (const int i,const int j)
	{
		return elements_on_host[i*n+j];
	}

	inline __host__ const float& operator() (const int i,const int j) const
	{
		return elements_on_host[i*n+j];
	}

	////copy data with four cases (CPU->CPU, GPU->CPU, GPU->GPU, CPU->GPU)
	__host__ Matrix& operator= (const Matrix& mtx)
	{
		if(on_host&&mtx.on_host){
			Resize_On_Host(mtx.m,mtx.n);
			elements_on_host=mtx.elements_on_host;
		}
		else if(on_host&&!mtx.on_host){
			Resize_On_Host(mtx.m,mtx.n);
			cudaMemcpy(&elements_on_host[0],mtx.elements_on_dev,m*n*sizeof(float),cudaMemcpyDeviceToHost);
		}
		else if(!on_host&&!mtx.on_host){
			Resize_On_Device(mtx.m,mtx.n);
			cudaMemcpy(elements_on_dev,mtx.elements_on_dev,mtx.m*n*sizeof(float),cudaMemcpyDeviceToDevice);
		}
		else if(!on_host&&mtx.on_host){
			Resize_On_Device(mtx.m,mtx.n);
			cudaMemcpy(elements_on_dev,&mtx.elements_on_host[0],m*n*sizeof(float),cudaMemcpyHostToDevice);
		}
		return *this;
	}

	////print matrix elements on screen
	__host__ friend ostream & operator << (ostream &out,const Matrix &mtx)
	{
		if(!mtx.on_host)
			cout<<"Print for matrix on device is not supported."<<endl;

		for(int i=0;i<mtx.m;i++){
			for(int j=0;j<mtx.n;j++){
				out<<mtx(i,j)<<", ";
			}
			out<<std::endl;
		}
		return out;
	}
};

//////////////////////////////////////////////////////////////////////////
////Your tasks start!

////This is a sample implementation without using any memory hierarchy
////The function calculates C=A*B, with dimA=[Am,An], dimB=[Bm,Bn], dimC=[Am,bn], and An=Bm
__global__ void Matrix_Multiplication_AB_Kernel_Poorman(const float* Ae,const float* Be,float* Ce,const int Am,const int An,const int Bn)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;

	float val=0.f;
	for(int k=0;k<An;k++)
		val+=Ae[i*An+k]*Be[k*Bn+j];
	Ce[i*Bn+j]=val;
} 

//////////////////////////////////////////////////////////////////////////
////Task 1: implement your fast matrix-matrix multiplication in the following kernel function.
////The function parameters are the same as the sample function:
////The function calculates the matrix multiplication, with C=A^T*B*A, A^T is the transpose of A, dimA=[Am,An], dimB=[Am,Am], and dimC=[An,An]
//////////////////////////////////////////////////////////////////////////

/*Your may want to declare your global variables here*/
__global__ void Matrix_Multiplication_AB_Kernel_Your_Version(const float* Ae,const float* Be,float* Ce,const int Am,const int An,const int Bn)
{
	/*Your implementation starts*/
	const int blockSize = 16;
	int tileRow = blockIdx.x;
	int tileCol = blockIdx.y;

	int row = threadIdx.x;
	int col = threadIdx.y;

	float* C_tile = &Ce[Bn * blockSize * tileRow + blockSize * tileCol];

	float Cvalue = 0;

	for (int m = 0; m < (An / blockSize); ++m) {
		 
		const float* A_tile = &Ae[An * tileRow * blockSize + m * blockSize];
		const float* B_tile = &Be[Bn * blockSize * m + tileCol *blockSize];

		__shared__ float A_s[blockSize][blockSize];
		__shared__ float B_s[blockSize][blockSize];

		A_s[row][col] = A_tile[row * An + col];
		B_s[row][col] = B_tile[row * Bn + col];
		__syncthreads();

		// Perform multiplication of the tile 
		for (int e = 0; e < blockSize; ++e) 
		{
		 Cvalue += A_s[row][e] * B_s[e][col];
		}
		__syncthreads();
	}
	C_tile[row * Bn + col] = Cvalue;

}

////This is a sample implementation without using any memory hierarchy
////The function calculates the matrix multiplication, with C=A^T*B*A, A^T is the transpose of A, dimA=[Am,An], dimB=[Am,Am], and dimC=[An,An]
__global__ void Matrix_Multiplication_ATBA_Kernel_Poorman(const float* Ae,const float* Be,float* Ce,const int Am,const int An)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	
	float val=0.f;
	for(int l=0;l<Am;l++)
		for(int k=0;k<Am;k++)
			val+=Ae[l*An+i]*Be[l*Am+k]*Ae[k*An+j];
	Ce[i*An+j]=val;
}

//////////////////////////////////////////////////////////////////////////
////Task 2: calculate the matrix multiplication in the following kernel function. 
////The function parameters are the same as the sample function:
////The function calculates the matrix multiplication, with C=A^T*B*A, A^T is the transpose of A, dimA=[Am,An], dimB=[Am,Am], and dimC=[An,An]
//////////////////////////////////////////////////////////////////////////

__global__ void Matrix_Multiplication_ATBA_Kernel_Your_Version(const float* Ae,const float* Be,float* Ce,const int Am,const int An)
{

	/*Your implementation starts*/
	const int blockSize = 16;
	int tileRow = blockIdx.x;
	int tileCol = blockIdx.y;

	int row = threadIdx.x;
	int col = threadIdx.y;

	float* C_tile = &Ce[Am * blockSize * tileRow + blockSize * tileCol];

	float Cvalue = 0;


	for (int m = 0; m < (An / blockSize); ++m) {
		 
		const float* A_tile = &Ae[An * tileRow * blockSize + m * blockSize];
		const float* A_tile_transpose = &Ae[Am * blockSize * m + tileCol *blockSize];
		const float* B_tile = &Be[Am * blockSize * m + tileCol *blockSize];

		__shared__ float A_s[blockSize][blockSize];
		__shared__ float B_s[blockSize][blockSize];
		__shared__ float A_s_tranpose[blockSize][blockSize];


		A_s[row][col] = A_tile[row * An + col];
		A_s_tranpose[row][col] = A_tile_transpose[col * Am + row];
		B_s[row][col] = B_tile[row * Am + col];
		__syncthreads();

		// Former setup:
		// for(int e=0;e<An;e++)
		// val += Ae[i*An+e]*Be[e*Bn+j];
		//
		// Cvalue += A_s[row][e] * B_s[e][col];


		// for(int l=0;l<Am;l++)
		// for(int k=0;k<Am;k++)
			// val+=Ae[l*An+i]*Be[l*Am+k]*Ae[k*An+j];

		for (int l = 0; l < blockSize; ++l)
		{
			for (int k = 0; k < blockSize; ++k) 
			{
				Cvalue += A_s_tranpose[row][l] * B_s[l][k] * A_s[k][col];
			}
		}
		__syncthreads();
	}
	C_tile[row * Am + col] = Cvalue;
	/*Your implementation ends*/
}

//////////////////////////////////////////////////////////////////////////
////Task 3:  calculate the Frobenius norm of a matrix
////The definition of F-norm for a matrix is square root of (the sum of squares of all the matrix elements), i.e., F=sqrt(sum_(A_ij^2))
////See the definition: https://mathworld.wolfram.com/FrobeniusNorm.html
//////////////////////////////////////////////////////////////////////////

////Please write your own kernel function here, and call it in the function Test_F_Norm_On_GPU to test its correctness and performance
/*Your implementation starts*/

__global__ void Test_F_Norm_On_GPU(const float *A, int An, int Am, int *norm)
{
	// extern __shared__ float shared_data[]; 
	// unsigned int 
	
}

/*Your implementation ends*/





////Congratulations, your tasks are all finished!
//////////////////////////////////////////////////////////////////////////


////Here are the test functions for your three kernel implementations

ofstream out;

__host__ void Test_Matrix_Multiplication_AB_On_GPU(const Matrix& A,const Matrix& B,Matrix& C)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	Matrix B_on_dev(B.m,B.n,false);
	B_on_dev=B;

	//// Allocate C in device memory
	Matrix C_on_dev(A_on_dev.m,B_on_dev.n,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	const int block_size=16;
	const int block_num_x=C.m/block_size;
	const int block_num_y=C.n/block_size;

	////TODO: this is a sample implementation. Comment it out to test your own code.
	// Matrix_Multiplication_AB_Kernel_Poorman<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
	// 	(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n,B_on_dev.n);

	////TODO: Uncomment this to test your own implementation
	////NOTICE: You do not have to use the block_size I specified here. You may customize the size of your grid and blocks for better performance.

	Matrix_Multiplication_AB_Kernel_Your_Version<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
		(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n,B_on_dev.n);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for matrix multiplication AB: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//// Transfer data back to CPU
	C=C_on_dev;

	out<<"T1: "<<gpu_time<<endl;
}

__host__ void Test_Matrix_Multiplication_ATBA_On_GPU(const Matrix& A,const Matrix& B,Matrix& C)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	Matrix B_on_dev(B.m,B.n,false);
	B_on_dev=B;
	
	//// Allocate C in device memory
	Matrix C_on_dev(A_on_dev.n,A_on_dev.n,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	const int block_size=16;
	const int block_num_x=C.m/block_size;
	const int block_num_y=C.n/block_size;

	////TODO: this is a sample implementation. Comment it out to test your own code.
	// Matrix_Multiplication_ATBA_Kernel_Poorman<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
	// 	(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);

	////TODO: Uncomment this to test your own implementation.
	////NOTICE: You do not have to use the block_size I specified here. You may customize the size of your grid and blocks for better performance.
	
	Matrix_Multiplication_ATBA_Kernel_Your_Version<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
		(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for matrix multiplication ATBA: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//// Transfer data back to CPU
	C=C_on_dev;

	out<<"T2: "<<gpu_time<<endl;
}

__host__ void Test_Matrix_F_Norm_On_GPU(const Matrix& A,/*result*/float& norm)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	////TODO: call the F norm kernel you implemented, and return the value to the passed-in variable norm

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for F norm: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	out<<"T3: "<<gpu_time<<endl;
}

int main()
{
	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_competition_1_matrix.dat";
	out.open(file_name.c_str());
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////
	////NOTICE: We may use a different set of parameters to evaluate your code.
	////So please test your functions with different size and initial values.
	//////////////////////////////////////////////////////////////////////////

	const int m=512;
	const int n=2048;
	const int p=1024;

	Matrix h_A(m,n);
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			h_A(i,j)=1.f;
		}
	}

	Matrix h_B(n,p);
	for(int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			h_B(i,j)=1.f;
		}
	}

	Matrix h_C(m,p);

	Matrix h_B2(m,m);
	for(int i=0;i<m;i++){
		for(int j=0;j<m;j++){
			h_B2(i,j)=1.f;
		}
	}

	Matrix h_C2(n,n);

	Test_Matrix_Multiplication_AB_On_GPU(h_A,h_B,h_C);
	cout<<"AB result: "<<h_C(h_C.m/2,h_C.n/2)<<endl;
	out<<"R1: "<<h_C(h_C.m/2,h_C.n/2)<<endl;

	Test_Matrix_Multiplication_ATBA_On_GPU(h_A,h_B2,h_C2);
	cout<<"ATBA result: "<<h_C2(h_C2.m/3,h_C2.n/3)<<endl;
	out<<"R2: "<<h_C2(h_C2.m/3,h_C2.n/3)<<endl;

	float f_norm=0.f;
	Test_Matrix_F_Norm_On_GPU(h_A,f_norm);
	cout<<"F-norm result: "<<f_norm<<endl;
	out<<"R3: "<<f_norm<<endl;

	return 0;
}