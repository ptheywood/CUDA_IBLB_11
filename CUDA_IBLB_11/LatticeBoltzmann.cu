#include <cmath>
#include <cstdlib>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LatticeBoltzmann.cuh"



__device__ const double C_S = 0.57735;
//__device__ const double G_PM = 1.0;
//__device__ const double TAU2 = 0.505556;
//__device__ const double RHO_0 = 1.;

__device__ const double c_l[15 * 3] =		//VELOCITY COMPONENTS
{
	0.,0.,0. ,
	1.,0.,0. , -1.,0.,0. , 0.,1.,0. , 0.,-1.,0. , 0.,0.,1. , 0.,0.,-1. ,
	1.,1.,1. , -1.,-1.,-1. , 1.,1.,-1. , -1.,-1.,1. , 1.,-1.,1. , -1.,1.,-1. , -1.,1.,1. , 1.,-1.,-1.
};

__device__ const double t[15] =					//WEIGHT VALUES
{
	2. / 9,
	1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9,
	1. / 72, 1. / 72, 1. / 72, 1. / 72, 1. / 72, 1. / 72, 1. / 72, 1. / 72
};

__device__ void DoubleAtomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
}

__global__ void equilibrium(const double * u, const double * rho, double * f0, const double * force, double * F, const int XDIM, const int YDIM, const int ZDIM, const double TAU)
{
	unsigned int i(0), j(0);

	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	double vec[3] = { 0.,0.,0. };

	int size = XDIM*YDIM*ZDIM;

	
	{
		j = threadnum;

		for (i = 0; i < 15; i++)
		{
			
			f0[15 * j + i] = rho[j] * t[i] * (1.
				+ (u[0 * size + j] * c_l[i * 3 + 0] + u[1 * size + j] * c_l[i * 3 + 1] + u[2 * size + j] * c_l[i * 3 + 2]) / (C_S*C_S)
				+ (u[0 * size + j] * c_l[i * 3 + 0] + u[1 * size + j] * c_l[i * 3 + 1] + u[2 * size + j] * c_l[i * 3 + 2]) * (u[0 * size + j] * c_l[i * 3 + 0] + u[1 * size + j] * c_l[i * 3 + 1] + u[2 * size + j] * c_l[i * 3 + 2]) / (2 * C_S*C_S*C_S*C_S)
				- (u[0 * size + j] * u[0 * size + j] + u[1 * size + j] * u[1 * size + j] + u[2 * size + j] * u[2 * size + j]) / (2 * C_S*C_S));
			

			vec[0] = (c_l[i * 3 + 0] - u[0 * size + j]) / (C_S*C_S) + (c_l[i * 3 + 0] * u[0 * size + j] + c_l[i * 3 + 1] * u[1 * size + j] + c_l[i * 3 + 1] * u[2 * size + j]) / (C_S*C_S*C_S*C_S) * c_l[i * 3 + 0];
			vec[1] = (c_l[i * 3 + 1] - u[1 * size + j]) / (C_S*C_S) + (c_l[i * 3 + 0] * u[0 * size + j] + c_l[i * 3 + 1] * u[1 * size + j] + c_l[i * 3 + 1] * u[2 * size + j]) / (C_S*C_S*C_S*C_S) * c_l[i * 3 + 1];
			vec[2] = (c_l[i * 3 + 2] - u[2 * size + j]) / (C_S*C_S) + (c_l[i * 3 + 0] * u[0 * size + j] + c_l[i * 3 + 1] * u[1 * size + j] + c_l[i * 3 + 1] * u[2 * size + j]) / (C_S*C_S*C_S*C_S) * c_l[i * 3 + 1];
/*
			vec[0] = c_l[i * 2 + 0] / (C_S*C_S) + ( (c_l[i * 2 + 0] * u[0 * size + j] + c_l[i * 2 + 1] * u[1 * size + j])*c_l[i * 2 + 0] - C_S*C_S*u[0 * size + j] ) / (C_S*C_S*C_S*C_S);
			vec[1] = c_l[i * 2 + 1] / (C_S*C_S) + ( (c_l[i * 2 + 0] * u[0 * size + j] + c_l[i * 2 + 1] * u[1 * size + j])*c_l[i * 2 + 1] - C_S*C_S*u[1 * size + j] ) / (C_S*C_S*C_S*C_S);
*/

			F[15 * j + i] = t[i] * (1. - 1. / (2. * TAU)) * (vec[0] * force[0 * size + j] + vec[1] * force[1 * size + j] + vec[2] * force[2 * size + j]);
			
		}
	}

	__syncthreads();
}

//f0[15*size]: equilibrium population density, f[15*size]: population density (pre-collision), f1[15*size]: population density (post-collision)
//F[15*size]: external forces, TAU: relaxation time (const), XDIM: x dimension (192), YDIM: y dimension (192), ZDIM: z dimension (const)

__global__ void collision(const double * f0, const double * f, double * f1, const double * F, const double TAU, const int XDIM, const int YDIM, const int ZDIM)
{
	unsigned int j(0), i(0);											//iterators for fluid node (j) and population density within each node (i)

	int size = XDIM*YDIM*ZDIM;											//total size of simulated fluid region

	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;				//individual thread number

	j = threadnum;														//one fluid node assigned to each thread

	for (i = 0 ; i < 15 ; i++)											//for each population density within a fluid node
	{
		f1[15 * j + i] = (1 - (1 / TAU))*f[15 * j + i] + (1 / TAU)*f0[15 * j + i] + F[15 * j + i];		//calculate population denity after 'collision' with equilibrium values and external forces

		// TRT method (no longer used)
		/*
			double rho_set = 1.;
		double u_set[2] = { 0.00004,0. };
		double u_s[2] = { 0.,0. };

		double omega_plus = 1. / TAU;
		double omega_minus = 1. / TAU2;


		double f_plus(0.), f_minus(0.), f0_plus(0.), f0_minus(0.);
			
			f1[9 * j + 0] = f[9 * j + 0] - omega_plus*(f[9 * j + 0] - f0[9 * j + 0]) + F[9 * j + 0];

			f_plus = (f[9 * j + 1] + f[9 * j + 3]) / 2.;
			f_minus = (f[9 * j + 1] - f[9 * j + 3]) / 2.;
			f0_plus = (f0[9 * j + 1] + f0[9 * j + 3]) / 2.;
			f0_minus = (f0[9 * j + 1] - f0[9 * j + 3]) / 2.;

			f1[9 * j + 1] = f[9 * j + 1] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus) + F[9 * j + 1];

			f_minus *= -1.;
			f0_minus *= -1.;

			f1[9 * j + 3] = f[9 * j + 3] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus) + F[9 * j + 3];

			f_plus = (f[9 * j + 2] + f[9 * j + 4]) / 2.;
			f_minus = (f[9 * j + 2] - f[9 * j + 4]) / 2.;
			f0_plus = (f0[9 * j + 2] + f0[9 * j + 4]) / 2.;
			f0_minus = (f0[9 * j + 2] - f0[9 * j + 4]) / 2.;

			f1[9 * j + 2] = f[9 * j + 2] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus) + F[9 * j + 2];

			f_minus *= -1.;
			f0_minus *= -1.;

			f1[9 * j + 4] = f[9 * j + 4] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus) + F[9 * j + 4];

			f_plus = (f[9 * j + 5] + f[9 * j + 7]) / 2.;
			f_minus = (f[9 * j + 5] - f[9 * j + 7]) / 2.;
			f0_plus = (f0[9 * j + 5] + f0[9 * j + 7]) / 2.;
			f0_minus = (f0[9 * j + 5] - f0[9 * j + 7]) / 2.;

			f1[9 * j + 5] = f[9 * j + 5] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus) + F[9 * j + 5];

			f_minus *= -1.;
			f0_minus *= -1.;

			f1[9 * j + 7] = f[9 * j + 7] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus) + F[9 * j + 7];

			f_plus = (f[9 * j + 6] + f[9 * j + 8]) / 2.;
			f_minus = (f[9 * j + 6] - f[9 * j + 8]) / 2.;
			f0_plus = (f0[9 * j + 6] + f0[9 * j + 8]) / 2.;
			f0_minus = (f0[9 * j + 6] - f0[9 * j + 8]) / 2.;

			f1[9 * j + 6] = f[9 * j + 6] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus) + F[9 * j + 6];

			f_minus *= -1.;
			f0_minus *= -1.;

			f1[9 * j + 8] = f[9 * j + 8] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus) + F[9 * j + 8];*/

	}

	//--------------------------------SHARED MEMORY USAGE (INCOMPLETE)------------------------------

	//
		//unsigned int k(0), n(0), m(0);
		//n = threadIdx.x;
		//
		//const int tile = 128;
		//const int tpoints = (tile - tile % 15) / 15;
		//int numtiles = (15 * size - (15 * size) % (tpoints * 15)) / (tpoints * 15);	//number of full tiles to populate the whole array of values
		//int excess = (15*size) % (tpoints * 15);	//number of values outside of full tiles
		//
		//__shared__ double sh_f0[tile];
		//__shared__ double sh_f[tile];
		//__shared__ double sh_F[tile];
		//
		//
		//	sh_f0[n] = 0.;
		//	sh_f[n] = 0.;
		//	sh_F[n] = 0.;
		//
		//
		//for (m = 0; m < numtiles; m++)		//iterate for each tile within the arrays
		//{
		//	__syncthreads();
		//
		//	
		//		sh_f0[n] = f0[m*tpoints*15 + n];
		//		sh_f[n] = f[m*tpoints*15 + n];
		//		sh_F[n] = F[m*tpoints*15 + n];
		//	
		//
		//	__syncthreads();
		//
		//	for (k = 0; k < tpoints; k++)
		//	{
		//		for (i = 0; i < 15; i++)
		//		{
		//			f1[15 * j + i] = (1 - (1 / TAU))*sh_f[15 * k + i] + (1 / TAU)*sh_f0[15 * k + i] + sh_F[15 * k + i];
		//
		//		}
		//	}
		//
		//	__syncthreads();
		//}
		//
		//
		//
		//if (n < excess)		//if there are excess values after the arrays have been split into tiles, and only execute for that many threads
		//{
		//	
		//		sh_f0[n] = f0[m*tpoints * 15 + n];		//take values from excess into shared memory
		//		sh_f[n] = f[m*tpoints * 15 + n];
		//		sh_F[n] = F[m*tpoints * 15 + n];
		//	
		//}
		//else
		//{
		//	
		//		sh_f0[n] = 0.;
		//		sh_f[n] = 0.;
		//		sh_F[n] = 0.;
		//	
		//}
		//
		//__syncthreads();
		//
		//for (k = 0; k < tpoints; k++)
		//{
		//	for (i = 0; i < 15; i++)
		//	{
		//		f1[15 * j + i] = (1 - (1 / TAU))*sh_f[15 * k + i] + (1 / TAU)*sh_f0[15 * k + i] + sh_F[15 * k + i];
		//
		//	}
		//}
		//
		//__syncthreads();

	//--------------------------------ZOU-HE VELOCITY BOUNDARY (NO LONGER USED)---------------------
	/*
		if (j%XDIM[0] == 0)										//LEFT
		{
		
		//rho_set = 1 / (1 - u_set[0])*(f[9 * j + 0] + f[9 * j + 2] + f[9 * j + 4] + 2 * (f[9 * j + 3] + f[9 * j + 6] + f[9 * j + 7]));
		rho_set = RHO_0;
		f1[9 * j + 1] = f[9 * j + 3] + (2./3.)*rho_set*u_set[0];

		f1[9 * j + 5] = f[9 * j + 7] - 0.5*(f[9 * j + 2] - f[9 * j + 4]) + 0.5*rho_set*u_set[1] + (1. / 6.)*rho_set*u_set[0];

		f1[9 * j + 8] = f[9 * j + 6] + 0.5*(f[9 * j + 2] - f[9 * j + 4]) - 0.5*rho_set*u_set[1] + (1. / 6.)*rho_set*u_set[0];
		}
		*/
	/*
		if (j % XDIM[0] == XDIM[0]-1 )										//RIGHT
		{
		rho_set = RHO_0;

		u_s[0] = 1. - (f[9 * j + 0] + f[9 * j + 2] + f[9 * j + 4] + 2. * (f[9 * j + 1] + f[9 * j + 5] + f[9 * j + 8]))/rho_set;

		u_s[1] = 0.;

		f1[9 * j + 3] = f[9 * j + 1] + (2. / 3.)*rho_set*u_s[0];

		f1[9 * j + 7] = f[9 * j + 5] - 0.5*(f[9 * j + 4] - f[9 * j + 2]) + 0.5*rho_set*u_s[1] + (1. / 6.)*rho_set*u_s[0];

		f1[9 * j + 6] = f[9 * j + 8] + 0.5*(f[9 * j + 4] - f[9 * j + 2]) - 0.5*rho_set*u_s[1] + (1. / 6.)*rho_set*u_s[0];
		}
		*/
	

	__syncthreads();
}

__global__ void streaming(const double * f1, double * f, int XDIM, int YDIM, int ZDIM, int it)
{
	
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0), k(0);
	unsigned int jstream(0);
	bool back(0), thru(0), done(0), slip(0), thrp(0), thrf(0);
	bool up(0), down(0), left(0), right(0), front(0), rear(0);

	int x(0), y(0), z(0);

	
	{
		j = threadnum;

		x = j%XDIM;
		y = ((j - (j % XDIM)) / XDIM) % YDIM;
		z = (j - j%(XDIM*YDIM)) / (XDIM*YDIM);

		//------------------------------------WALL CONDITIONS------------------------------------------------

		up = 0;
		down = 0;
		left = 0;
		right = 0;
		front = 0;
		rear = 0;

		if (y == YDIM - 1) up = 1;
		if (y == 0) down = 1;
		if (x == 0) left = 1;
		if (x == XDIM - 1) right = 1;
		if (z == 0) rear = 1;
		if (z == ZDIM - 1) front = 1;



		for (i = 0; i < 15; i++)
		{
			//cout << i << endl;

			back = 0;
			thru = 0;
			done = 0;
			slip = 0;
			thrp = 0;
			thrf = 0;
			k = i;

			
			//printf("\n%d\n", y);

			//---------------------------------------------------MID GRID NON-SLIP BOUNDARY------------------------------

			if (down || up || left || right || front || rear) 
			{
				switch (i)
				{
				case 0: break;

				case 1:
					if (right) { thru = 1;}
					break;

				case 2:
					if (left) { thru = 1; }
					break;

				case 3:
					if (up) { slip = 1; }
					break;

				case 4:
					if (down) { back = 1; }
					break;

				case 5:
					if (front) { thrf = 1; }
					break;

				case 6:
					if (rear) { thrf = 1; }
					break;

				case 7:
					if (up) { slip = 1; }
					else if (right && front) { jstream = j - XDIM*YDIM*(ZDIM-1) + 1; done = 1; }
					else if (right) { thru = 1; }
					else if (front) { thrf = 1; }
					break;

				case 8:
					if (down) { back = 1; }
					else if (left && rear) { jstream = j + XDIM*YDIM*(ZDIM - 1) - 1; done = 1; }
					else if (left) { thru = 1; }
					else if (rear) { thrf = 1; }
					break;

				case 9:
					if (up) { slip = 1; }
					else if (right && rear) { jstream = j + XDIM*YDIM*(ZDIM - 1) + 1; done = 1; }
					else if (right) { thru = 1; }
					else if (rear) { thrf = 1; }
					break;

				case 10:
					if (down) { back = 1; }
					else if (left && front) { jstream = j - XDIM*YDIM*(ZDIM - 1) - 1; done = 1; }
					else if (left) { thru = 1; }
					else if (front) { thrf = 1; }
					break;

				case 11:
					if (down) { back = 1; }
					else if (right && front) { jstream = j - XDIM*(YDIM*(ZDIM - 1) + 2) + 1; done = 1; }
					else if (right) { thru = 1; }
					else if (front) { thrf = 1; }
					break;

				case 12:
					if (up) { slip = 1; }
					else if (left && rear) { jstream = j + XDIM*(YDIM*(ZDIM - 1) + 2) - 1; done = 1; }
					else if (left) { thru = 1; }
					else if (rear) { thrf = 1; }
					break;

				case 13:
					if (up) { slip = 1; }
					else if (left && front) { jstream = j - XDIM*(YDIM*(ZDIM - 1) - 2) - 1; done = 1; }
					else if (left) { thru = 1; }
					else if (front) { thrf = 1; }
					break;

				case 14:
					if (down) { back = 1; }
					else if (right && rear) { jstream = j + XDIM*(YDIM*(ZDIM - 1) - 2) + 1; done = 1; }
					else if (right) { thru = 1; }
					else if (rear) { thrf = 1; }
					break;

				}

			}

			//--------------------------------------------------STREAMING CALCULATIONS-------------------------------

			if (back && !done)
			{
				jstream = j; //BACK STREAM

				if (i == 1) k = 2;
				if (i == 2) k = 1;
				if (i == 3) k = 4;
				if (i == 4) k = 3;
				if (i == 5) k = 6;
				if (i == 6) k = 5;
				if (i == 7) k = 8;
				if (i == 8) k = 7;
				if (i == 9) k = 10;
				if (i == 10) k = 9;
				if (i == 11) k = 12;
				if (i == 12) k = 11;
				if (i == 13) k = 14;
				if (i == 14) k = 13;
			}
			else if (slip && !done)
			{
				jstream = j; //SLIP STREAM

				if (i == 1) k = 1;
				if (i == 2) k = 2;
				if (i == 3) k = 4;
				if (i == 4) k = 3;
				if (i == 5) k = 5;
				if (i == 6) k = 6;
				if (i == 7) k = 11;
				if (i == 8) k = 12;
				if (i == 9) k = 14;
				if (i == 10) k = 13;
				if (i == 11) k = 7;
				if (i == 12) k = 8;
				if (i == 13) k = 10;
				if (i == 14) k = 9;
			}
			else if (thru && !done)
			{
				jstream = j - (XDIM-1)*c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; //THROUGH STREAM SIDE

				k = i;
			}
			else if (thrp && !done)
			{
				jstream = j + c_l[i * 3 + 0] - XDIM*(YDIM - 1)*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; //THROUGH STREAM UP

				k = i;

				printf("\nTHRP!!\n");
			}
			else if (thrf && !done)
			{
				jstream = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] - XDIM*YDIM*(ZDIM - 1)*c_l[i * 3 + 2]; //THROUGH STREAM FRONT

				k = i;

			}
			else if (!done)
			{
				jstream = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; //NORMAL STREAM

				k = i;
			}

			
			f[15 * jstream + k] = f1[15 * j + i];								//STREAM TO ADJACENT CELL IN DIRECTION OF MOVEMENT
		}
	}

	__syncthreads();

}

__global__ void macro(const double * f_P, const double * f_M, double * rho_P, double * rho_M, double * rho, double * u, double * u_M, int XDIM, int YDIM, int ZDIM)
{
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0);


	{
		j = threadnum;

		int size = XDIM*YDIM*ZDIM;

		rho[j] = 0;

		rho_P[j] = 0.;
		rho_M[j] = 0.;

		double momentum[3] = { 0.,0.,0. };

		double M_flux[3] = { 0.,0.,0. };

		u[0 * size + j] = 0.;
		u[1 * size + j] = 0.;
		u[2 * size + j] = 0.;

		u_M[0 * size + j] = 0.;
		u_M[1 * size + j] = 0.;
		u_M[2 * size + j] = 0.;

		for (i = 0; i < 15; i++)
		{
			rho_P[j] += f_P[15 * j + i];
			rho_M[j] += f_M[15 * j + i];

			momentum[0] += 1.*c_l[i * 3 + 0] * (f_P[15 * j + i] + f_M[15 * j + i]);
			momentum[1] += 1.*c_l[i * 3 + 1] * (f_P[15 * j + i] + f_M[15 * j + i]);
			momentum[2] += 1.*c_l[i * 3 + 2] * (f_P[15 * j + i] + f_M[15 * j + i]);

			M_flux[0] += 1.*c_l[i * 3 + 0] * (f_M[15 * j + i]);
			M_flux[1] += 1.*c_l[i * 3 + 1] * (f_M[15 * j + i]);
			M_flux[2] += 1.*c_l[i * 3 + 2] * (f_M[15 * j + i]);
		}

		rho[j] = rho_P[j] + rho_M[j];

		u[0 * size + j] = 1.*(momentum[0]) / (1.*rho[j]);
		u[1 * size + j] = 1.*(momentum[1]) / (1.*rho[j]);
		u[2 * size + j] = 1.*(momentum[2]) / (1.*rho[j]);

		u_M[0 * size + j] = 1.*(M_flux[0]) / (1.*rho_M[j]);
		u_M[1 * size + j] = 1.*(M_flux[1]) / (1.*rho_M[j]);
		u_M[2 * size + j] = 1.*(M_flux[2]) / (1.*rho_M[j]);

		
	}

	__syncthreads();
}

__global__ void binaryforces(const double * rho_P, const double * rho_M, const double * rho, const double * f_P, const double * f_M, double * force_P, double * force_M, double * force_E, double * u, double * u_M, int XDIM, int YDIM, int ZDIM, const float G_PM, int it)
{
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0);

	const int size = XDIM*YDIM*ZDIM;

	int next(0);

	j = threadnum;

	double temp[6] = { 0., 0., 0., 0., 0., 0. };


	double momentum[3] = { 0., 0., 0. };

	double psi_P = 1. - exp(-1.*rho_P[j]);
	double psi_M = 1. - exp(-1.*rho_M[j]);
	double psi_Pn;
	double psi_Mn;

	bool up(0), down(0), left(0), right(0), front(0), rear(0), epi(0), air(0), done(0);

	//double force_GP_P[2] = { 0.,0. };
	//double force_GP_M[2] = { 0.,0. };

	double force_PE[3] = { 0.,0.,0. };
	double force_PA[3] = { 0.,0.,0. };
	double force_ME[3] = { 0.,0.,0. };
	double force_MA[3] = { 0.,0.,0. };

	force_P[0 * size + j] = 0.;
	force_P[1 * size + j] = 0.;
	force_P[2 * size + j] = 0.;
	force_M[0 * size + j] = 0.;
	force_M[1 * size + j] = 0.;
	force_M[2 * size + j] = 0.;

	double G_PE = 0;
	double G_PA = 0.;	//1.
	double G_ME = 0.;	//1.
	double G_MA = 0; 

	float mu = 0.3; //0.3
	float t_el = 100000000.; // 100 x T equivelant to 6 seconds: in range of real gel
	double Delta_u[3] = { 0.,0.,0. };

	int x = j%XDIM;
	int y = ((j - (j % XDIM)) / XDIM) % YDIM;
	int z = (j - j%(XDIM*YDIM)) / (XDIM*YDIM);

	if (y == YDIM - 1) up = 1;
	if (y == 0) down = 1;
	if (x == 0) left = 1;
	if (x == XDIM - 1) right = 1;
	if (z == 0) rear = 1;
	if (z == ZDIM - 1) front = 1;

	
	for (i = 1; i < 15; i++)
	{
		epi = 0;
		air = 0;
		done = 0;
		next = 0;

		if (down || up || left || right || front || rear)
		{
			switch (i)
			{
			case 0: break;

			case 1:
				if (right)
				{
					next = j - (XDIM - 1);
					done = 1;
				}
				break;

			case 2:
				if (left)
				{
					next = j + (XDIM - 1);
					done = 1;
				}
				break;

			case 3:
				if (up)
				{
					air = 1;
					//next = j - XDIM * (YDIM - 1);
					done = 1;
				}
				break;

			case 4:
				if (down)
				{
					epi = 1;
					//next = j + (YDIM - 1)*XDIM;
					done = 1;
				}
				break;

			case 5:
				if (front)
				{
					next = j - XDIM*YDIM*(ZDIM - 1);
					done = 1;
				}

				break;

			case 6:
				if (rear)
				{
					next = j + XDIM*YDIM*(ZDIM - 1);
					done = 1;
				}
				
				break;

			case 7:
				if (up) { air = 1; done = 1; }
				else if (right && front) { next = j - XDIM*YDIM*(ZDIM - 1) + 1; done = 1; }
				else if (right) { next = j - (XDIM - 1)*c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; done = 1; }
				else if (front) { next = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] - XDIM*YDIM*(ZDIM - 1)*c_l[i * 3 + 2]; done = 1; }
				break;

			case 8:
				if (down) { epi = 1; done = 1; }
				else if (left && rear) { next = j + XDIM*YDIM*(ZDIM - 1) - 1; done = 1; }
				else if (left) { next = j - (XDIM - 1)*c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; done = 1; }
				else if (rear) { next = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] - XDIM*YDIM*(ZDIM - 1)*c_l[i * 3 + 2]; done = 1; }
				break;

			case 9:
				if (up) { air = 1; done = 1; }
				else if (right && rear) { next = j + XDIM*YDIM*(ZDIM - 1) + 1; done = 1; }
				else if (right) { next = j - (XDIM - 1)*c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; done = 1; }
				else if (rear) { next = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] - XDIM*YDIM*(ZDIM - 1)*c_l[i * 3 + 2]; done = 1; }
				break;

			case 10:
				if (down) { epi = 1; done = 1; }
				else if (left && front) { next = j - XDIM*YDIM*(ZDIM - 1) - 1; done = 1; }
				else if (left) { next = j - (XDIM - 1)*c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; done = 1; }
				else if (front) { next = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] - XDIM*YDIM*(ZDIM - 1)*c_l[i * 3 + 2]; done = 1; }
				break;

			case 11:
				if (down) { epi = 1; done = 1; }
				else if (right && front) { next = j - XDIM*(YDIM*(ZDIM - 1) + 2) + 1; done = 1; }
				else if (right) { next = j - (XDIM - 1)*c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; done = 1; }
				else if (front) { next = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] - XDIM*YDIM*(ZDIM - 1)*c_l[i * 3 + 2]; done = 1; }
				break;

			case 12:
				if (up) { air = 1; done = 1; }
				else if (left && rear) { next = j + XDIM*(YDIM*(ZDIM - 1) + 2) - 1; done = 1; }
				else if (left) { next = j - (XDIM - 1)*c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; done = 1; }
				else if (rear) { next = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] - XDIM*YDIM*(ZDIM - 1)*c_l[i * 3 + 2]; done = 1; }
				break;

			case 13:
				if (up) { air = 1; done = 1; }
				else if (left && front) { next = j - XDIM*(YDIM*(ZDIM - 1) - 2) - 1; done = 1; }
				else if (left) { next = j - (XDIM - 1)*c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; done = 1; }
				else if (front) { next = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] - XDIM*YDIM*(ZDIM - 1)*c_l[i * 3 + 2]; done = 1; }
				break;

			case 14:
				if (down) { epi = 1; done = 1; }
				else if (right && rear) { next = j + XDIM*(YDIM*(ZDIM - 1) - 2) + 1; done = 1; }
				else if (right) { next = j - (XDIM - 1)*c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2]; done = 1; }
				else if (rear) { next = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] - XDIM*YDIM*(ZDIM - 1)*c_l[i * 3 + 2]; done = 1; }
				break;

			}
		}

		if (!done)
		{
			next = j + c_l[i * 3 + 0] + XDIM*c_l[i * 3 + 1] + XDIM*YDIM*c_l[i * 3 + 2];		
		}

		if (epi)
		{
			force_PE[0] += -1. * psi_P * G_PE * t[i] * c_l[i * 3 + 0];
			force_PE[1] += -1. * psi_P * G_PE * t[i] * c_l[i * 3 + 1];
			force_PE[2] += -1. * psi_P * G_PE * t[i] * c_l[i * 3 + 2];

			force_ME[0] += -1. * psi_M * G_ME * t[i] * c_l[i * 3 + 0];
			force_ME[1] += -1. * psi_M * G_ME * t[i] * c_l[i * 3 + 1];
			force_ME[2] += -1. * psi_M * G_ME * t[i] * c_l[i * 3 + 2];
		}
		else if (air)
		{
			force_PA[0] += -1. * psi_P * G_PA * t[i] * c_l[i * 3 + 0];
			force_PA[1] += -1. * psi_P * G_PA * t[i] * c_l[i * 3 + 1];
			force_PA[2] += -1. * psi_P * G_PA * t[i] * c_l[i * 3 + 2];

			force_MA[0] += -1. * psi_M * G_MA * t[i] * c_l[i * 3 + 0];
			force_MA[1] += -1. * psi_M * G_MA * t[i] * c_l[i * 3 + 1];
			force_MA[2] += -1. * psi_M * G_MA * t[i] * c_l[i * 3 + 2];
		}
		else
		{
			psi_Pn = 1. - exp(-1.*rho_P[next]);
			psi_Mn = 1. - exp(-1.*rho_M[next]);

			temp[0] += 1.* t[i] * psi_Mn * c_l[i * 3 + 0];
			temp[1] += 1.* t[i] * psi_Mn * c_l[i * 3 + 1];
			temp[2] += 1.* t[i] * psi_Mn * c_l[i * 3 + 2];
			temp[3] += 1.* t[i] * psi_Pn * c_l[i * 3 + 0];
			temp[4] += 1.* t[i] * psi_Pn * c_l[i * 3 + 1];
			temp[5] += 1.* t[i] * psi_Pn * c_l[i * 3 + 2];

			Delta_u[0] += 1. * t[i] * (u_M[0 * size + next] - u_M[0 * size + j]);
			Delta_u[1] += 1. * t[i] * (u_M[1 * size + next] - u_M[1 * size + j]);
			Delta_u[2] += 1. * t[i] * (u_M[2 * size + next] - u_M[2 * size + j]);
		}

		//if (j == XDIM-1) printf("wall? %d -> %d \n", i, wall);

	}

	temp[0] *= -1. * psi_P * G_PM;
	temp[1] *= -1. * psi_P * G_PM;
	temp[2] *= -1. * psi_P * G_PM;
	temp[3] *= -1. * psi_M * G_PM;
	temp[4] *= -1. * psi_M * G_PM;
	temp[5] *= -1. * psi_M * G_PM;

	force_E[0 * size + j] = force_E[0 * size + j] * (1. - 1. / t_el) + 2. * mu / (t_el * C_S * C_S) * Delta_u[0];
	force_E[1 * size + j] = force_E[1 * size + j] * (1. - 1. / t_el) + 2. * mu / (t_el * C_S * C_S) * Delta_u[1];
	force_E[2 * size + j] = force_E[2 * size + j] * (1. - 1. / t_el) + 2. * mu / (t_el * C_S * C_S) * Delta_u[2];

	force_P[0 * size + j] += 1. * temp[0] + force_PE[0] + force_PA[0];
	force_P[1 * size + j] += 1. * temp[1] + force_PE[1] + force_PA[1];
	force_P[2 * size + j] += 1. * temp[2] + force_PE[2] + force_PA[2];
	force_M[0 * size + j] += 1. * temp[3] + force_ME[0] + force_MA[0] + force_E[0 * size + j];
	force_M[1 * size + j] += 1. * temp[4] + force_ME[1] + force_MA[1] + force_E[1 * size + j];
	force_M[2 * size + j] += 1. * temp[5] + force_ME[2] + force_MA[2] + force_E[2 * size + j];

	__syncthreads();

	u[0 * size + j] = 0.;
	u[1 * size + j] = 0.;
	u[2 * size + j] = 0.;

	momentum[0] = 0.;
	momentum[1] = 0.;
	momentum[2] = 0.;

	
		for (i = 0; i < 15; i++)
		{
			momentum[0] += 1.*c_l[i * 3 + 0] * (f_P[15 * j + i] + f_M[15 * j + i]);
			momentum[1] += 1.*c_l[i * 3 + 1] * (f_P[15 * j + i] + f_M[15 * j + i]);
			momentum[2] += 1.*c_l[i * 3 + 2] * (f_P[15 * j + i] + f_M[15 * j + i]);
		}
	
	u[0 * size + j] = 1.*(momentum[0] + 0.5*(force_P[0 * size + j] + force_M[0 * size + j])) / (1.*rho[j]);
	u[1 * size + j] = 1.*(momentum[1] + 0.5*(force_P[1 * size + j] + force_M[1 * size + j])) / (1.*rho[j]);
	u[2 * size + j] = 1.*(momentum[2] + 0.5*(force_P[2 * size + j] + force_M[2 * size + j])) / (1.*rho[j]);

	//if ((j - j%XDIM) / XDIM == 0) u[0 * size + j] = 0.001 * cos(it * 2.* 3.14159 / 100.);  //oscillating lower wall for elastic test

	__syncthreads();

	//if (j == size / 2) printf("\n u_y[mid] = %f \n", f_P[9 * j + 2]);


}

__global__ void forces(const double * rho_P, const double * rho_M, const double * rho, const double * f_P, const double * f_M, const double * force, double * force_P, double * force_M, double * u, double * Q, double * Q_P, double * Q_M, int XDIM, int YDIM, int ZDIM)
{
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0);

	const int size = XDIM*YDIM*ZDIM;

	j = threadnum;

	double momentum[3];
	double spd(0.);

	double P_flux = 0.;
	double M_flux = 0.;

	double psi_P = 1. - exp(-1.*rho_P[j]);
	double psi_M = 1. - exp(-1.*rho_M[j]);

	force_P[0 * size + j] += 1.*(rho_P[j] / (1.*rho[j])) * force[0 * size + j];
	force_P[1 * size + j] += 1.*(rho_P[j] / (1.*rho[j])) * force[1 * size + j];
	force_P[2 * size + j] += 1.*(rho_P[j] / (1.*rho[j])) * 0.;
	force_M[0 * size + j] += 1.*(rho_M[j] / (1.*rho[j])) * force[0 * size + j];
	force_M[1 * size + j] += 1.*(rho_M[j] / (1.*rho[j])) * force[1 * size + j];
	force_M[2 * size + j] += 1.*(rho_M[j] / (1.*rho[j])) * 0.;

	__syncthreads();

	u[0 * size + j] = 0.;
	u[1 * size + j] = 0.;
	u[2 * size + j] = 0.;

	momentum[0] = 0.;
	momentum[1] = 0.;
	momentum[2] = 0.;

	for (i = 0; i < 15; i++)
	{
		momentum[0] += 1.*c_l[i * 3 + 0] * (f_P[15 * j + i] + f_M[15 * j + i]);
		momentum[1] += 1.*c_l[i * 3 + 1] * (f_P[15 * j + i] + f_M[15 * j + i]);
		momentum[2] += 1.*c_l[i * 3 + 2] * (f_P[15 * j + i] + f_M[15 * j + i]);

		P_flux += 1.*c_l[i * 3 + 0] * f_P[15 * j + i];
		M_flux += 1.*c_l[i * 3 + 0] * f_M[15 * j + i];
	}

	u[0 * size + j] = (momentum[0] + 0.5*(force_P[0 * size + j] + force_M[0 * size + j])) / rho[j];
	u[1 * size + j] = (momentum[1] + 0.5*(force_P[1 * size + j] + force_M[1 * size + j])) / rho[j];
	u[2 * size + j] = (momentum[2] + 0.5*(force_P[2 * size + j] + force_M[2 * size + j])) / rho[j];

	__syncthreads();


	if (j%XDIM == XDIM - 5)
	{
		spd = u[0 * size + j] / (YDIM*ZDIM);
		P_flux /= (YDIM*ZDIM);
		M_flux /= (YDIM*ZDIM);

		DoubleAtomicAdd(Q, spd);
		DoubleAtomicAdd(Q_P, P_flux);
		DoubleAtomicAdd(Q_M, M_flux);
	}

	__syncthreads();

	//if (j == XDIM*1.5) printf("\n force_x[mid] = %f \n", force[0 * size + j]);

}