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
			double one_over_dis_cube=1.0/pow(sqrt(dis_squared+epsilon_squared),3);
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
__device__ void findAccel(const double ipos_x, const double ipos_y, const double ipos_z, //// Body compared to (constant throughout kernel call)
						  const double jpos_x, const double jpos_y, const double jpos_z, //// Body comparing to  
						  const double jmass, 							  				//// Mass of ipos
						  const double epsilon_squared
						  double* ai_delta_x, double* ai_delta_y, double* ai_delta_z)
{

	// Compute the Denominator of the Acceleration Update
	double r_x = jpos_x - ipos_x;
	double r_y = jpos_y - ipos_y;
	double r_z = jpos_z - ipos_z;
	double r2 = r_x * rx + ry * ry + rz * rz + epsilon_squared;
	double r6 = r2 * r2 * r2
	double r6_inverted = 1.0 / sqrt(r6);
	double directionless_ai = r6_inverted * jmass


	// Compute the change in acceleration:
	ai_delta_x = r_x * directionless_ai;
	ai_delta_y = r_y * directionless_ai;
	ai_delta_z = r_z * directionless_ai;
	

}

// Computes Velocity and Positino given Acceleration for a single body 
// Given time step, position, velocity, and acceleration for that body 
__device__ void findVP(const int dt, 
						double pos_x, double pos_y, double pos_z, 
						double vel_x, double vel_y, double vel_z, 
						double acl_x, double acl_y, double acl_z, 
						)
{
	
}

// Tiling Function 
// For each body/thread 
	// Tiles body arrays into manageable chunks -> shared memory 
	// Calls findForce as the tile progresses through shared memory 
	


	// Computes final position with findVP per body 

__global__ void tileForceBodies(double* pos_x, double* pos_y, double* pos_z, double* mass) 
{

	// blockDim x blockDim shared memory
	// [x0, x1, x2, x3, ..., xn, y0, y1, y2, ..., yn]
	extern __shared__ double[] bodyData;

	const int x_idx = 0;
	const int y_idx = blockDim.x;
	const int z_idx = blockDim.x * 2;
	const int w_idx = blockDim.x * 3;

	global_tid = blockIdx.x * blockDim.x + threadIdx.x
	local_tid = threadIdx.x;


	// This thread's information:
	double ipos_x = pos_x[global_tid];
	double ipos_y = pos_y[global_tid];
	double ipos_z = pos_z[global_tid];
	double ai_x = 0; 
	double ai_y = 0;
	double ai_z = 0;



	// Load shared memory 
	for(int i = 0; i < N, i+=blockDim.x) { // divides into pxp chunks
		// load position values for blockDim elements into shared memory:
		bodyData[x_idx + local_tid] = pos_x[i + local_tid];
		bodyData[y_idx + local_tid] = pos_y[i + local_tid];
		bodyData[z_idx + local_tid] = pos_z[i + local_tid];
		bodyData[w_idx + local_tid] = mass[i + local_tid];
		__syncthreads();

		// Calculate interactions with body i 
		for(int j = i; j < i + blockDim.x; j++) {
			jpos_x = bodyData[x_idx + j];
			jpos_y = bodyData[y_idx + j];
			jpos_z = bodyData[z_idx + j];
			jmass = bodyData[w_idx + j];

			findAccel(const double ipos_x, const double ipos_y, const double ipos_z,
					  const double jpos_x, const double jpos_y, const double jpos_z,
					  const double jmass, 
					  const double epsilon_squared, 
					  double* ai_x, double* ai_y, double* ai_z); 

		}


	}

}

thread_count = 128; 

k = N/thread_count + 1 ; 

tileForceBodies<<k,thread_count>>

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
const unsigned int grid_size=4;					////assuming particles are initialized on a background grid
const unsigned int particle_n=pow(grid_size,3);	////assuming each grid cell has one particle at the beginning

__host__ void Test_N_Body_Simulation()
{
	////initialize position, velocity, acceleration, and mass
	
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

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////

	out<<"R0: "<<pos_x[particle_n/2]<<" " <<pos_y[particle_n/2]<<" " <<pos_z[particle_n/2]<<endl;
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