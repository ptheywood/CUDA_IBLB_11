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

__device__ const double c_l[9 * 2] =		//VELOCITY COMPONENTS
{
	0.,0. ,
	1.,0. , 0.,1. , -1.,0. , 0.,-1. ,
	1.,1. , -1.,1. , -1.,-1. , 1.,-1.
};

__device__ const double t[9] =					//WEIGHT VALUES
{
	4. / 9,
	1. / 9, 1. / 9, 1. / 9, 1. / 9,
	1. / 36, 1. / 36, 1. / 36, 1. / 36
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

__global__ void equilibrium(const double * u, const double * rho, double * f0, const double * force, double * F, const int XDIM, const int YDIM, const double TAU)
{
	unsigned int i(0), j(0);

	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	double vec[2] = { 0.,0. };

	int size = XDIM*YDIM;

	
	{
		j = threadnum;

		for (i = 0; i < 9; i++)
		{
			
			f0[9 * j + i] = rho[j] * t[i] * (1.
				+ (u[0 * size + j] * c_l[2 * i + 0] + u[1 * size + j] * c_l[2 * i + 1]) / (C_S*C_S)
				+ (u[0 * size + j] * c_l[2 * i + 0] + u[1 * size + j] * c_l[2 * i + 1]) * (u[0 * size + j] * c_l[2 * i + 0] + u[1 * size + j] * c_l[2 * i + 1]) / (2 * C_S*C_S*C_S*C_S)
				- (u[0 * size + j] * u[0 * size + j] + u[1 * size + j] * u[1 * size + j]) / (2 * C_S*C_S));
			

			vec[0] = (c_l[i * 2 + 0] - u[0 * size + j]) / (C_S*C_S) + (c_l[i * 2 + 0] * u[0 * size + j] + c_l[i * 2 + 1] * u[1 * size + j]) / (C_S*C_S*C_S*C_S) * c_l[i * 2 + 0];
			vec[1] = (c_l[i * 2 + 1] - u[1 * size + j]) / (C_S*C_S) + (c_l[i * 2 + 0] * u[0 * size + j] + c_l[i * 2 + 1] * u[1 * size + j]) / (C_S*C_S*C_S*C_S) * c_l[i * 2 + 1];
/*
			vec[0] = c_l[i * 2 + 0] / (C_S*C_S) + ( (c_l[i * 2 + 0] * u[0 * size + j] + c_l[i * 2 + 1] * u[1 * size + j])*c_l[i * 2 + 0] - C_S*C_S*u[0 * size + j] ) / (C_S*C_S*C_S*C_S);
			vec[1] = c_l[i * 2 + 1] / (C_S*C_S) + ( (c_l[i * 2 + 0] * u[0 * size + j] + c_l[i * 2 + 1] * u[1 * size + j])*c_l[i * 2 + 1] - C_S*C_S*u[1 * size + j] ) / (C_S*C_S*C_S*C_S);
*/

			F[9 * j + i] = t[i] * (1. - 1. / (2. * TAU)) * (vec[0] * force[0 * size + j] + vec[1] * force[1 * size + j]);
			
		}
	}

	__syncthreads();
}

__global__ void collision(const double * f0, const double * f, double * f1, const double * F, double TAU, int XDIM, int YDIM)
{
	unsigned int j(0), i(0);

	//double rho_set = 1.;
	//double u_set[2] = { 0.00004,0. };
	//double u_s[2] = { 0.,0. };

	//double omega_plus = 1. / TAU;
	//double omega_minus = 1. / TAU2;


	//double f_plus(0.), f_minus(0.), f0_plus(0.), f0_minus(0.);

	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	{
		j = threadnum;

		for (i = 0 ; i < 9 ; i++)
		{
			f1[9 * j + i] = (1 - (1 / TAU))*f[9 * j + i] + (1 / TAU)*f0[9 * j + i] + F[j * 9 + i];


			// TRT method
			/*f1[9 * j + 0] = f[9 * j + 0] - omega_plus*(f[9 * j + 0] - f0[9 * j + 0]) + F[9 * j + 0];

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

		//--------------------------------ZOU-HE VELOCITY BOUNDARY-------------------------
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
	}

	__syncthreads();
}

__global__ void streaming(const double * f1, double * f, int XDIM, int YDIM, int it)
{
	
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0), k(0);
	unsigned int jstream(0);
	bool back(0), thru(0), done(0), slip(0), thrp(0);
	bool up(0), down(0), left(0), right(0);

	int x(0), y(0);

	
	{
		j = threadnum;

		x = j%XDIM;
		y = (j - j%XDIM) / XDIM;

		//------------------------------------WALL CONDITIONS------------------------------------------------

		up = 0;
		down = 0;
		left = 0;
		right = 0;

		if (y == YDIM - 1) up = 1;
		if (y == 0) down = 1;
		if (x == 0) left = 1;
		if (x == XDIM - 1) right = 1;

		for (i = 0; i < 9; i++)
		{
			//cout << i << endl;

			back = 0;
			thru = 0;
			done = 0;
			slip = 0;
			thrp = 0;
			k = i;


			//---------------------------------------------------MID GRID NON-SLIP BOUNDARY------------------------------

			if (down || up || left || right)
			{
				switch (i)
				{
				case 0: break;

				case 1:
					if (right) { thru = 1;}
					break;
				case 2:
					if (up) { /*thrp = 1;*/ /*slip = 1;*/ back = 1; }
					break;

				case 3:
					if (left) { thru = 1; }
					break;

				case 4:
					if (down) { /*thrp = 1;*/ back = 1; }
					break;

				case 5:
					//if (up && right) { jstream = 0; done = 1; }
					if (up) { /*thrp = 1;*/ /*slip = 1;*/ back = 1; }
					else if (right) { thru = 1; }
					break;

				case 6:
					//if (up && left) { jstream = XDIM - 1; done = 1; }
					if (up) { /*thrp = 1;*/ /*slip = 1;*/ back = 1; }
					else if (left) { thru = 1; }
					break;

				case 7:
					//if (down && left) { jstream = XDIM*YDIM - 1; done = 1; }
					if (down) { /*thrp = 1;*/ back = 1; }
					else if (left) { thru = 1; }
					break;

				case 8:
					//if (down && right) { jstream = XDIM*YDIM - XDIM; done = 1; }
					if (down) { /*thrp = 1;*/ back = 1; }
					else if (right) { thru = 1; }
					break;
				}

			}

			//--------------------------------------------------STREAMING CALCULATIONS-------------------------------

			if (back && !done)
			{
				jstream = j; //BACK STREAM

				if (i == 1) k = 3;
				if (i == 2) k = 4;
				if (i == 3) k = 1;
				if (i == 4) k = 2;
				if (i == 5) k = 7;
				if (i == 6) k = 8;
				if (i == 7) k = 5;
				if (i == 8) k = 6;
			}
			else if (slip && !done)
			{
				jstream = j; //SLIP STREAM

				if (i == 1) k = 1;
				if (i == 2) k = 4;
				if (i == 3) k = 3;
				if (i == 4) k = 2;
				if (i == 5) k = 8;
				if (i == 6) k = 7;
				if (i == 7) k = 6;
				if (i == 8) k = 5;
			}
			else if (thru && !done)
			{
				jstream = j - (XDIM-1)*c_l[i * 2 + 0] + XDIM*c_l[i * 2 + 1]; //THROUGH STREAM

				k = i;
			}
			else if (thrp && !done)
			{
				jstream = j + c_l[i * 2 + 0] - XDIM*(YDIM - 1)*c_l[i * 2 + 1]; //THROUGH STREAM

				k = i;

				printf("\nTHRP!!\n");
			}
			else if (!done)
			{
				jstream = j + c_l[i * 2 + 0] + XDIM*c_l[i * 2 + 1]; //NORMAL STREAM

				k = i;
			}

			//if (back && down && i == 1) f[9 * jstream + k] = (f1[9 * j + 1] + f1[9 * j + 3] ) * cos(it / 100000.);
			//else if (back && down && i == 3) f[9 * jstream + k] = (f1[9 * j + 1] + f1[9 * j + 3]) * (1. - cos(it / 100000.));
			f[9 * jstream + k] = f1[9 * j + i];								//STREAM TO ADJACENT CELL IN DIRECTION OF MOVEMENT
		}
	}

	__syncthreads();

}

__global__ void macro(const double * f_P, const double * f_M, double * rho_P, double * rho_M, double * rho, double * u, double * u_M, int XDIM, int YDIM)
{
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0);


	{
		j = threadnum;

		int size = XDIM*YDIM;

		rho[j] = 0;

		rho_P[j] = 0.;
		rho_M[j] = 0.;

		double momentum[2] = { 0.,0. };

		double M_flux[2] = { 0.,0. };

		u[0 * size + j] = 0.;
		u[1 * size + j] = 0.;

		u_M[0 * size + j] = 0.;
		u_M[1 * size + j] = 0.;

		for (i = 0; i < 9; i++)
		{
			rho_P[j] += f_P[9 * j + i];
			rho_M[j] += f_M[9 * j + i];

			momentum[0] += 1.*c_l[i * 2 + 0] * (f_P[9 * j + i] + f_M[9 * j + i]);
			momentum[1] += 1.*c_l[i * 2 + 1] * (f_P[9 * j + i] + f_M[9 * j + i]);

			M_flux[0] += 1.*c_l[i * 2 + 0] * (f_M[9 * j + i]);
			M_flux[1] += 1.*c_l[i * 2 + 1] * (f_M[9 * j + i]);
		}

		rho[j] = rho_P[j] + rho_M[j];

		u[0 * size + j] = 1.*(momentum[0]) / (1.*rho[j]);
		u[1 * size + j] = 1.*(momentum[1]) / (1.*rho[j]);

		u_M[0 * size + j] = 1.*(M_flux[0]) / (1.*rho_M[j]);
		u_M[1 * size + j] = 1.*(M_flux[1]) / (1.*rho_M[j]);

		
	}

	__syncthreads();
}

__global__ void binaryforces(const double * rho_P, const double * rho_M, const double * rho, const double * f_P, const double * f_M, double * force_P, double * force_M, double * force_E, double * u, double * u_M, int XDIM, int YDIM, const float G_PM, int it)
{
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0);

	const int size = XDIM*YDIM;

	int next(0);

	j = threadnum;

	double temp[4];

	temp[0] = 0.;
	temp[1] = 0.;
	temp[2] = 0.;
	temp[3] = 0.;

	double momentum[2];

	double psi_P = 1. - exp(-1.*rho_P[j]);
	double psi_M = 1. - exp(-1.*rho_M[j]);
	double psi_Pn;
	double psi_Mn;

	bool up(0), down(0), left(0), right(0), epi(0), air(0), done(0);

	//double force_GP_P[2] = { 0.,0. };
	//double force_GP_M[2] = { 0.,0. };

	double force_PE[2] = { 0.,0. };
	double force_PA[2] = { 0.,0. };
	double force_ME[2] = { 0.,0. };
	double force_MA[2] = { 0.,0. };

	force_P[0 * size + j] = 0.;
	force_P[1 * size + j] = 0.;
	force_M[0 * size + j] = 0.;
	force_M[1 * size + j] = 0.;

	double G_PE = 0;
	double G_PA = 0.;	//1.
	double G_ME = 0.;	//1.
	double G_MA = 0; 

	float mu = 0.;
	float t_el = 10000000.;
	double Delta_u[2] = { 0.,0. };

	int x = j%XDIM;
	int y = (j - j%XDIM) / XDIM;

	if (y == YDIM - 1) up = 1;
	if (y == 0) down = 1;
	if (x == 0) left = 1;
	if (x == XDIM - 1) right = 1;


	for (i = 1; i < 9; i++)
	{
		epi = 0;
		air = 0;
		done = 0;
		next = 0;

		if (down || up || left || right)
		{
			switch (i)
			{
			case 0: break;

			case 1:
				if (right)
				{
					next = j - 1 * (XDIM - 1);
					done = 1;
				}
				break;

			case 2:
				if (up)
				{
					air = 1;
					//next = j - (YDIM - 1)*XDIM;
					done = 1;
				}
				break;
			case 3:

				if (left)
				{
					next = j + 1 * (XDIM - 1);
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

				if (up)
				{
					air = 1;
					//next = j - (YDIM - 1)*XDIM;
					done = 1;
				}
				else if (right)
				{
					next = j + 1;
					done = 1;
				}

				break;
			case 6:

				if (up)
				{
					air = 1;
					//next = j - (YDIM - 1)*XDIM;
					done = 1;
				}
				else if (left)
				{
					next = j + (2 * XDIM - 1);
					done = 1;
				}

				break;
			case 7:

				if (down)
				{
					epi = 1;
					//next = j + (YDIM - 1)*XDIM;
					done = 1;
				}
				else if (left)
				{
					next = j - 1;
					done = 1;
				}

				break;
			case 8:

				if (down)
				{
					epi = 1;
					//next = j + (YDIM - 1)*XDIM;
					done = 1;
				}
				else if (right)
				{
					next = j - 2 * XDIM + 1;
					done = 1;
				}

				break;
			}
		}

		if (!done)
		{
			next = j + c_l[i * 2 + 0] + XDIM*c_l[i * 2 + 1];		//checked, all correct
		}

		if (epi)
		{
			force_PE[0] += -1. * psi_P * G_PE * t[i] * c_l[i * 2 + 0];
			force_PE[1] += -1. * psi_P * G_PE * t[i] * c_l[i * 2 + 1];

			force_ME[0] += -1. * psi_M * G_ME * t[i] * c_l[i * 2 + 0];
			force_ME[1] += -1. * psi_M * G_ME * t[i] * c_l[i * 2 + 1];
		}
		else if (air)
		{
			force_PA[0] += -1. * psi_P * G_PA * t[i] * c_l[i * 2 + 0];
			force_PA[1] += -1. * psi_P * G_PA * t[i] * c_l[i * 2 + 1];

			force_MA[0] += -1. * psi_M * G_MA * t[i] * c_l[i * 2 + 0];
			force_MA[1] += -1. * psi_M * G_MA * t[i] * c_l[i * 2 + 1];
		}
		else
		{
			psi_Pn = 1. - exp(-1.*rho_P[next]);
			psi_Mn = 1. - exp(-1.*rho_M[next]);

			temp[0] += 1.* t[i] * psi_Mn * c_l[i * 2 + 0];
			temp[1] += 1.* t[i] * psi_Mn * c_l[i * 2 + 1];
			temp[2] += 1.* t[i] * psi_Pn * c_l[i * 2 + 0];
			temp[3] += 1.* t[i] * psi_Pn * c_l[i * 2 + 1];

			Delta_u[0] += 1. * t[i] * (u_M[0 * size + next] - u_M[0 * size + j]);
			Delta_u[1] += 1. * t[i] * (u_M[1 * size + next] - u_M[1 * size + j]);
		}

		//if (j == XDIM-1) printf("wall? %d -> %d \n", i, wall);

	}

	temp[0] *= -1. * psi_P * G_PM;
	temp[1] *= -1. * psi_P * G_PM;
	temp[2] *= -1. * psi_M * G_PM;
	temp[3] *= -1. * psi_M * G_PM;

	force_E[0 * size + j] = force_E[0 * size + j] * (1. - 1. / t_el) + 2. * mu / (t_el * C_S * C_S) * Delta_u[0];
	force_E[1 * size + j] = force_E[1 * size + j] * (1. - 1. / t_el) + 2. * mu / (t_el * C_S * C_S) * Delta_u[1];

	force_P[0 * size + j] += 1. * temp[0] + force_PE[0] + force_PA[0];
	force_P[1 * size + j] += 1. * temp[1] + force_PE[1] + force_PA[1];
	force_M[0 * size + j] += 1. * temp[2] + force_ME[0] + force_MA[0] + force_E[0 * size + j];
	force_M[1 * size + j] += 1. * temp[3] + force_ME[1] + force_MA[1] + force_E[1 * size + j];

	__syncthreads();

	u[0 * size + j] = 0.;
	u[1 * size + j] = 0.;

	momentum[0] = 0.;
	momentum[1] = 0.;

	
		for (i = 0; i < 9; i++)
		{
			momentum[0] += 1.*c_l[i * 2 + 0] * (f_P[9 * j + i] + f_M[9 * j + i]);
			momentum[1] += 1.*c_l[i * 2 + 1] * (f_P[9 * j + i] + f_M[9 * j + i]);
		}
	
	u[0 * size + j] = 1.*(momentum[0] + 0.5*(force_P[0 * size + j] + force_M[0 * size + j])) / (1.*rho[j]);
	u[1 * size + j] = 1.*(momentum[1] + 0.5*(force_P[1 * size + j] + force_M[1 * size + j])) / (1.*rho[j]);

	if ((j - j%XDIM) / XDIM == 0) u[0 * size + j] = 0.001 * cos(it * 2.* 3.14159 / 10000.);

	__syncthreads();

	//if (j == size / 2) printf("\n u_y[mid] = %f \n", f_P[9 * j + 2]);


}

__global__ void forces(const double * rho_P, const double * rho_M, const double * rho, const double * f_P, const double * f_M, const double * force, double * force_P, double * force_M, double * u, double * Q, double * Q_P, double * Q_M, int XDIM, int YDIM)
{
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0);

	const int size = XDIM*YDIM;

	j = threadnum;

	double momentum[2];
	double spd(0.);

	double P_flux = 0.;
	double M_flux = 0.;

	double psi_P = 1. - exp(-1.*rho_P[j]);
	double psi_M = 1. - exp(-1.*rho_M[j]);

	force_P[0 * size + j] += 1.*(rho_P[j] / (1.*rho[j])) * force[0 * size + j];
	force_P[1 * size + j] += 1.*(rho_P[j] / (1.*rho[j])) * force[1 * size + j];
	force_M[0 * size + j] += 1.*(rho_M[j] / (1.*rho[j])) * force[0 * size + j];
	force_M[1 * size + j] += 1.*(rho_M[j] / (1.*rho[j])) * force[1 * size + j];

	__syncthreads();

	u[0 * size + j] = 0.;
	u[1 * size + j] = 0.;

	momentum[0] = 0.;
	momentum[1] = 0.;

	for (i = 0; i < 9; i++)
	{
		momentum[0] += 1.*c_l[i * 2 + 0] * (f_P[9 * j + i] + f_M[9 * j + i]);
		momentum[1] += 1.*c_l[i * 2 + 1] * (f_P[9 * j + i] + f_M[9 * j + i]);

		P_flux += 1.*c_l[i * 2 + 0] * f_P[9 * j + i];
		M_flux += 1.*c_l[i * 2 + 0] * f_M[9 * j + i];
	}

	u[0 * size + j] = (momentum[0] + 0.5*(force_P[0 * size + j] + force_M[0 * size + j])) / rho[j];
	u[1 * size + j] = (momentum[1] + 0.5*(force_P[1 * size + j] + force_M[1 * size + j])) / rho[j];

	__syncthreads();


	if (j%XDIM == XDIM - 5)
	{
		spd = u[0 * size + j] / YDIM;
		P_flux /= YDIM;
		M_flux /= YDIM;

		DoubleAtomicAdd(Q, spd);
		DoubleAtomicAdd(Q_P, P_flux);
		DoubleAtomicAdd(Q_M, M_flux);
	}

	__syncthreads();

	//if (j == XDIM*1.5) printf("\n force_x[mid] = %f \n", force[0 * size + j]);

}