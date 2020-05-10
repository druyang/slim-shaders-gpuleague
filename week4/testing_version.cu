//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 2: n-body simulation
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define NUM_TESTS 200; 

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team="Slim_Shaders";
	std::string author_1="Andrw_Yang";
	std::string author_2="Matthew_Kenney";
};

//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here

// Compute Acceleration from Force interaction between two bodies  
__device__ double3 findAccel(const double4 ipos, const double4 jpos, //// Body comparing to  
						  const double epsilon_squared, double3 ai)
{
	// ipos -> position (and mass) of body i
	// jpos -> position (and mass) of body j
	// epsilon_sqared -> softening factor
	// ai -> acceleration of body i to update

	// Compute the Denominator of the Acceleration Update
	double rx = jpos.x - ipos.x;
	double ry = jpos.y - ipos.y;
	double rz = jpos.z - ipos.z;
	double r2 = rx * rx + ry * ry + rz * rz;
	double r_new = r2+epsilon_squared; 
	double r_6 = r_new * r_new * r_new; 
	double directionless_ai = 1.0 / sqrt(r_6); 

	// Compute the change in acceleration:
	ai.x += rx * directionless_ai * jpos.w;
	ai.y += ry * directionless_ai * jpos.w;
	ai.z += rz * directionless_ai * jpos.w;

	return ai;

}
// Computes Velocity and Position given Acceleration for a single body 
// Given time step, position, velocity, and acceleration for that body 
__device__ double3 findV(double3 vel, double3 acc, const double dt)
{
	// update velocity 
	vel.x += acc.x * dt;
	vel.y += acc.y * dt;
	vel.z += acc.z * dt;

	return vel;

}
__device__ double4 findP(double4 pos, double3 vel, const double dt)
{
	// update position 
	pos.x += vel.x * dt;
	pos.y += vel.y * dt;
	pos.z += vel.z * dt;

	return pos;

}


// Tiling Function 
// For each body/thread 
	// Tiles body arrays into manageable chunks -> shared memory 
	// Calls findForce as the tile progresses through shared memory 
	


	// Computes final position with findVP per body 

__global__ void tileForceBodies(double4* pos, double3 *vel, double3 *acc, 
								const double epsilon_squared, 
								const double dt, 
								const int particle_n) 
{
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	int local_tid = threadIdx.x;

	if (global_tid < particle_n) {

		// This thread's information:
		double4 this_pos = pos[global_tid]; // w (mass), x, y, z
		double3 this_vel = vel[global_tid]; // current body's velocity
		double3 this_acc; // current acceleration (set via N-body computation)
		this_acc.x = 0.0;
		this_acc.y = 0.0;
		this_acc.z = 0.0;


		// 1 x blockDim shared memory
		extern __shared__ double4 sharedMem[];
		double4* body_data = &sharedMem[0]; // p * sizeof(double4)
		double3* acc_data = &sharedMem[blockDim.x]; // p * p * sizeof(double3)
		// acc due to j=0, acc due to j=1, ...

		// Load shared memory 
		#pragma unroll 4
		for(int i = 0; i < particle_n; i+=blockDim.x) { // divides particles into N/blockDim chunks

			// load position values for blockDim particles into shared memory:
			body_data[local_tid] = pos[i + local_tid]; // move blockDim slots ahead on each outer loop execution
			__syncthreads();

			// Calculate interactions between current body & all bodies j in the domain j ∈ [i, i + blockDim) 
			#pragma unroll 8
			for(int j = 0; j < blockDim.x; j++) {
				double4 jpos = body_data[j];
				// atomicAdd(&this_acc, findAccelChange(this_pos, jpos, epsilon_squared, this_acc));
				this_acc = findAccel(this_pos, jpos, epsilon_squared, this_acc);
			}
			__syncthreads();
		}



		// Find position and velocity 
		this_vel = findV(this_vel, this_acc, dt);
		this_pos = findP(this_pos, this_vel, dt);

		// write back to global memory:
		acc[global_tid] = this_acc;
		vel[global_tid] = this_vel;
		pos[global_tid] = this_pos;
	}
}



////Your implementations end here
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
////Test function for n-body simulator
ofstream out;

//////////////////////////////////////////////////////////////////////////
////Please do not change the values below
const double dt=0.001;							////time step
const int time_step_num=10;						////number of time steps
const double epsilon=1e-2;						////epsilon added in the denominator to avoid 0-division when calculating the gravitational force
const double epsilon_squared=epsilon*epsilon;	////epsilon squared

////We use grid_size=4 to help you debug your code, change it to a bigger number (e.g., 16, 32, etc.) to test the performance of your GPU code
const unsigned int grid_size= 16;					////assuming particles are initialized on a background grid
const unsigned int particle_n=pow(grid_size,3);	////assuming each grid cell has one particle at the beginning

// Thread Count is min of particle_n and 512 (so as not to spawn excess threads in the case of a small number of bodies)
const unsigned int thread_count = min(particle_n, 128);
//const unsigned int thread_count = 128;


__host__ void Test_N_Body_Simulation()
{
	////initialize position, velocity, acceleration, and mass
	printf("Using %d threads per block\n", thread_count);
	printf("Using %d blocks\n\n", (int)ceil(double(particle_n)/double(thread_count)));
	
	double* pos_x=new double[particle_n];
	double* pos_y=new double[particle_n];
	double* pos_z=new double[particle_n];
	////initialize particle positions as the cell centers on a background grid
	double dx=1.0/(double)grid_size;
	for(unsigned int k=0;k<grid_size;k++){
		for(unsigned int j=0;j<grid_size;j++){
			for(unsigned int i=0;i<grid_size;i++){
				unsigned int index=k*grid_size*grid_size+j*grid_size+i;
				pos_x[index]=dx*(double)i;
				pos_y[index]=dx*(double)j;
				pos_z[index]=dx*(double)k;
			}
		}
	}

	double* vel_x=new double[particle_n];
	memset(vel_x,0x00,particle_n*sizeof(double));
	double* vel_y=new double[particle_n];
	memset(vel_y,0x00,particle_n*sizeof(double));
	double* vel_z=new double[particle_n];
	memset(vel_z,0x00,particle_n*sizeof(double));

	double* acl_x=new double[particle_n];
	memset(acl_x,0x00,particle_n*sizeof(double));
	double* acl_y=new double[particle_n];
	memset(acl_y,0x00,particle_n*sizeof(double));
	double* acl_z=new double[particle_n];
	memset(acl_z,0x00,particle_n*sizeof(double));

	double* mass=new double[particle_n];
	for(int i=0;i<particle_n;i++) {
		mass[i] = 100.0;
	}


	//////////////////////////////////////////////////////////////////////////
	// Creating double values like CPU and moving to GPU 
	float avg_time = 0.0f; 

	for(int k = 0; k < 200; k++)
	{
	double4* pos_host = new double4[particle_n]; 
	double3* vel_host= new double3[particle_n]; 
	double3* acl_host = new double3[particle_n];

	// Set position and mass data in pos_gpu 
	for(unsigned int k=0;k<grid_size;k++){
		for(unsigned int j=0;j<grid_size;j++){
			for(unsigned int i=0;i<grid_size;i++){
				unsigned int index=k*grid_size*grid_size+j*grid_size+i;
				pos_host[index].x = dx*(double)i; 
				pos_host[index].y = dx*(double)j; 
				pos_host[index].z = dx*(double)k; 
			}
		}
	}

	for(int i=0;i<particle_n;i++) {
		pos_host[i].w=100.0;
	}
	
	// set velocity and acceleration vectors to 0 
	for(int i=0; i<particle_n; i++){
		vel_host[i].x = 0;
		vel_host[i].y = 0;
		vel_host[i].z = 0;
	}

	// Copy vectors over to GPU  
	double4* pos_gpu; 
	double3* vel_gpu; 
	double3* acl_gpu; 

	cudaMalloc((void**)&pos_gpu, particle_n * sizeof(double4)); 
	cudaMalloc((void**)&vel_gpu, particle_n * sizeof(double3)); 
	cudaMalloc((void**)&acl_gpu, particle_n * sizeof(double3)); 

	cudaMemcpy(pos_gpu, pos_host, particle_n*sizeof(double4), cudaMemcpyHostToDevice);
	cudaMemcpy(vel_gpu, vel_host, particle_n*sizeof(double3), cudaMemcpyHostToDevice); 

	//////////////////////////////////////////////////////////////////////////
	////Your implementation: n-body simulator on GPU


		cudaEvent_t start,end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		float gpu_time=0.0f;
		cudaDeviceSynchronize();
		cudaEventRecord(start);

		//////////////////////////////////////////////////////////////////////////
		////TODO 2: Your GPU functions are called here
		////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
		////The final positions should be stored in the same place as the CPU n-body function, i.e., pos_x, pos_y, pos_z
		////The correctness of your simulation will be evaluated by comparing the results (positions) with the results calculated by the default CPU implementations

		//////////////////////////////////////////////////////////////////////////
		int num_blocks = ceil((double)particle_n/(double)thread_count);
		
		cout<<"\nTotal number of particles: "<<particle_n<<endl;
		cout<<"Tracking the motion of particle "<<particle_n/2<<endl;

		// Step through time 
		for(int i=0;i<time_step_num;i++){

			tileForceBodies<<<num_blocks, thread_count, thread_count * sizeof(double4)>>>(pos_gpu, vel_gpu, acl_gpu, epsilon_squared, dt, particle_n);
			cudaMemcpy(pos_host, pos_gpu, particle_n*sizeof(double4), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cout<<"pos on timestep "<<i<<": "<<pos_host[particle_n/2].x<<", "<<pos_host[particle_n/2].y<<", "<<pos_host[particle_n/2].z<<endl;
		}

		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&gpu_time,start,end);
		printf("\nGPU runtime: %.4f ms\n",gpu_time);
		cudaEventDestroy(start);
		cudaEventDestroy(end);
		avg_time += gpu_time; 
	//////////////////////////////////////////////////////////////////////////

	out<<"R0: "<<pos_host[particle_n/2].x<<" " <<pos_host[particle_n/2].y<<" " <<pos_host[particle_n/2].z<<endl;
	out<<"T1: "<<gpu_time<<endl;
	}

	avg_time = avg_time/NUM_TESTS; 
	cout<<"average time"<<avg_time<<endl; 


}

int main()
{

	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_competition_2_nbody.dat";
	out.open(file_name.c_str());
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	Test_N_Body_Simulation();

	return 0;
}