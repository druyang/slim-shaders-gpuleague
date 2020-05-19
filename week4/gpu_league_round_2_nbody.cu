//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 2: n-body simulation
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
	std::string team="Slim_Shaders";
	std::string author_1="Andrw_Yang";
	std::string author_2="Matthew_Kenney";
};

//////////////////////////////////////////////////////////////////////////
////Here is a sample function implemented on CPU for n-body simulation.

__host__ void N_Body_Simulation_CPU_Poorman(double* pos_x,double* pos_y,double* pos_z,		////position array
											double* vel_x,double* vel_y,double* vel_z,		////velocity array
											double* acl_x,double* acl_y,double* acl_z,		////acceleration array
											const double* mass,								////mass array
											const int n,									////number of particles
											const double dt,								////timestep
											const double epsilon_squared)					////epsilon to avoid 0-denominator
{		
	////Step 1: set particle accelerations to be zero
	memset(acl_x,0x00,sizeof(double)*n);
	memset(acl_y,0x00,sizeof(double)*n);
	memset(acl_z,0x00,sizeof(double)*n);

	////Step 2: traverse all particle pairs and accumulate gravitational forces for each particle from pairwise interactions
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			////skip calculating force for itself
			if(i==j) continue;

			////r_ij=x_j-x_i
			double rx=pos_x[j]-pos_x[i];
			double ry=pos_y[j]-pos_y[i];
			double rz=pos_z[j]-pos_z[i];

			////a_ij=m_j*r_ij/(r+epsilon)^3, 
			////noticing that we ignore the gravitational coefficient (assuming G=1)
			double dis_squared=rx*rx+ry*ry+rz*rz;
			double dis_test = dis_squared+epsilon_squared; 
			double dis_6 = dis_test * dis_test * dis_test; 
			double one_over_dis_cube=rsqrt(dis_6); 
			double ax=mass[j]*rx*one_over_dis_cube;
			double ay=mass[j]*ry*one_over_dis_cube;
			double az=mass[j]*rz*one_over_dis_cube;

			////accumulate the force to the particle
			acl_x[i]+=ax;
			acl_y[i]+=ay;
			acl_z[i]+=az;
		}
	}

	////Step 3: explicit time integration to update the velocity and position of each particle
	for(int i=0;i<n;i++){
		////v_{t+1}=v_{t}+a_{t}*dt
		vel_x[i]+=acl_x[i]*dt;
		vel_y[i]+=acl_y[i]*dt;
		vel_z[i]+=acl_z[i]*dt;

		////x_{t+1}=x_{t}+v_{t}*dt
		pos_x[i]+=vel_x[i]*dt;
		pos_y[i]+=vel_y[i]*dt;
		pos_z[i]+=vel_z[i]*dt;
	}
}


//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here

// Compute Acceleration from Force interaction between two bodies  
__device__ double3 findAccel(const double4 ipos, const double4 jpos, //// Body comparing to  
						  const double epsilon_squared, double3 ai)
{
	// ipos -> position (and mass) of body i
	// jpos -> position (and mass) of body j
	// epsilon_squared -> softening factor
	// ai -> acceleration of body i to update

	// Compute the Denominator of the Acceleration Update
	double rx = jpos.x - ipos.x;
	double ry = jpos.y - ipos.y;
	double rz = jpos.z - ipos.z;
	double r2 = rx * rx + ry * ry + rz * rz + epsilon_squared;
	double r_6 = r2 * r2 * r2; 
	double directionless_ai = jpos.w * rsqrt(r_6); 

	// Compute the change in acceleration:
	ai.x += rx * directionless_ai;
	ai.y += ry * directionless_ai;
	ai.z += rz * directionless_ai;

	return ai;

}


// Computes Velocity given Acceleration for a single body 
// Given time step, velocity, and acceleration for that body 
__device__ double3 findV(double3 vel, double3 acc, const double dt)
{
	// update velocity 
	vel.x += acc.x * dt;
	vel.y += acc.y * dt;
	vel.z += acc.z * dt;

	return vel;

}

// Computes the acceleration changes to all bodies for a given time step.
// Implements a tiling approach in order to achieve shared memory speedups.
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
		extern __shared__ double4 bodyData[];
	
		// Load shared memory 
		#pragma unroll 4
		for(int i = 0; i < particle_n; i+=blockDim.x) { // divides particles into N/blockDim chunks

			// load position values for blockDim particles into shared memory:
			bodyData[local_tid] = pos[i + local_tid]; // move blockDim slots ahead on each outer loop execution
			__syncthreads();

			// Calculate interactions between current body & all bodies j in the domain j ∈ [i, i + blockDim) 
			#pragma unroll 32
			for(int j = 0; j < blockDim.x; j++) {
				double4 jpos = bodyData[j];
				this_acc = findAccel(this_pos, jpos, epsilon_squared, this_acc);
			}
			__syncthreads();
		}

		// Find velocity 
		this_vel = findV(this_vel, this_acc, dt);

		// write back to global memory:
		acc[global_tid] = this_acc;
		vel[global_tid] = this_vel;
	}
}

// Kernel Function to update the positions of all bodies once acceleration update has finished
__global__ void updatePositions(double4* pos, double3* vel, const double dt) {

	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	pos[global_tid].x += vel[global_tid].x * dt;
	pos[global_tid].y += vel[global_tid].y * dt;
	pos[global_tid].z += vel[global_tid].z * dt;

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
const unsigned int grid_size=20;					////assuming particles are initialized on a background grid
const unsigned int particle_n=pow(grid_size,3);	////assuming each grid cell has one particle at the beginning

// Thread Count is min of particle_n and 32 (so as not to spawn excess threads in the case of a small number of bodies)
const unsigned int thread_count = min(particle_n, 32);

__host__ void Test_N_Body_Simulation()
{
	////initialize position, velocity, acceleration, and mass
	//printf("Using %d threads per block\n", thread_count);
	//printf("Using %d blocks\n\n", (int)ceil(double(particle_n)/double(thread_count)));
	
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
	////Default implementation: n-body simulation on CPU
	////Comment the CPU implementation out when you test large-scale examples
	auto cpu_start=chrono::system_clock::now();
	cout<<"Total number of particles: "<<particle_n<<endl;
	cout<<"Tracking the motion of particle "<<particle_n/2<<endl;

	for(int i=0;i<time_step_num;i++){

		N_Body_Simulation_CPU_Poorman(pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acl_x,acl_y,acl_z,mass,particle_n,dt,epsilon_squared);
		cout<<"pos on timestep "<<i<<": "<<pos_x[particle_n/2]<<", "<<pos_y[particle_n/2]<<", "<<pos_z[particle_n/2]<<endl;

	}

	auto cpu_end=chrono::system_clock::now();
	chrono::duration<double> cpu_time=cpu_end-cpu_start;
	cout<<"CPU runtime: "<<cpu_time.count()*1000.<<" ms."<<endl;

	//////////////////////////////////////////////////////////////////////////
	// Creating double values like CPU and moving to GPU 

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
	cout<<"Print statements disabled "<<endl;

	// Step through time 
	for(int i=0;i<time_step_num;i++){

		// Here, we synchronize global memory before updating positions to avoid
		// Read-after-write conflicts for large values of particle_n
		tileForceBodies<<<num_blocks, thread_count, thread_count * sizeof(double4)>>>
			(pos_gpu, vel_gpu, acl_gpu, epsilon_squared, dt, particle_n);
	
		// Synchronize and write to global memory
		cudaDeviceSynchronize();
		updatePositions<<<num_blocks, thread_count>>>(pos_gpu, vel_gpu, dt);
		cudaDeviceSynchronize();
		
		// Print Results to console (comment out to test performance)
		cudaMemcpy(pos_host, pos_gpu, particle_n*sizeof(double4), cudaMemcpyDeviceToHost);
		cout<<"pos on timestep "<<i<<": "<<pos_host[particle_n/2].x<<", "<<pos_host[particle_n/2].y<<", "<<pos_host[particle_n/2].z<<endl;
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//////////////////////////////////////////////////////////////////////////
	// One Memcpy at the end of all kernel calls to return data to the host:
	cudaMemcpy(pos_host, pos_gpu, particle_n*sizeof(double4), cudaMemcpyDeviceToHost);

	// NOTE: Since we used our own double4 to store the values of our particles, we altered the write statement here
	// to reflect the way that we stored our values.
	out<<"R0: "<<pos_host[particle_n/2].x<<" " <<pos_host[particle_n/2].y<<" " <<pos_host[particle_n/2].z<<endl;
	out<<"T1: "<<gpu_time<<endl;
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
