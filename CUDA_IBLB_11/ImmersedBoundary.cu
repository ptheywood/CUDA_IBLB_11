#include <cmath>
#include <cstdlib>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ImmersedBoundary.cuh"
#include "device_functions.h"

#define PI 3.14159
//__device__ const double RHO_0 = 1.;
//__device__ const double C_S = 0.57735;

__constant__ double c_l[9 * 2] =		//VELOCITY COMPONENTS
{
	0.,0. ,
	1.,0. , 0.,1. , -1.,0. , 0.,-1. ,
	1.,1. , -1.,1. , -1.,-1. , 1.,-1.
};

__device__ double sqroot(double square)
{
	const double MINDIFF = 0.1;
	double root = square / 0.75, last, diff = 1;
	if (square <= 0) return 0;
	do {
		last = root;
		root = (root + square / root) / 2;
		diff = root - last;
	} while (diff > MINDIFF || diff < -MINDIFF);
	return root;
}

__device__ double delta(const double & xs, const double & ys, const int & x, const int & y)
{
	double deltax(0.), deltay(0.), delta(0.);

	double dx = abs(x - xs);
	double dy = abs(y - ys);

	double a(0.), b(0.), d(0.);
	int c(0);

	if (dx <= 1.5)
	{
		if (dx <= 0.5)
		{
			//deltax = (1. / 3.)*(1. + sqrt(-3. * dx*dx + 1.));
			a = 0.33333;
			b = 1.;
			c = 1;
			d = dx;
		}
		else //deltax = (1. / 6.)*(5. - 3. * dx - sqrt(-3. * (1. - dx)*(1. - dx) + 1.));
		{
			a = 0.16667;
			b = 5.-3.*dx;
			c = -1;
			d = 1-dx;
		}
	}

	deltax = a*(b + c*sqrt(-3.*d*d + 1));

	a = 0.;
	b = 0.;
	c = 0;
	d = 0.;

	if (dy <= 1.5)
	{
		if (dy <= 0.5)
		{
			//deltay = (1. / 3.)*(1. + sqrt(-3. * dy*dy + 1.));
			a = 0.33333;
			b = 1.;
			c = 1;
			d = dy;
		}
		else //deltay = (1. / 6.)*(5. - 3. * dy - sqrt(-3. * (1. - dy)*(1. - dy) + 1.));
		{
			a = 0.16667;
			b = 5. - 3.*dy;
			c = -1;
			d = 1 - dy;
		}
	}

	deltay = a*(b + c*sqrt(-3.*d*d + 1));

	delta = deltax * deltay;

	return delta;
}

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

__global__ void interpolate(const double * rho, const double * u, const int Ns, const double * u_s, double * F_s, const double * s, const int XDIM)
{

	int i(0), j(0), k(0), x0(0), y0(0), x(0), y(0);

	double xs(0.), ys(0.), del(0.);


	k = blockIdx.x*blockDim.x + threadIdx.x;


	{
		F_s[2 * k + 0] = 0.;
		F_s[2 * k + 1] = 0.;

		xs = s[k * 2 + 0];
		ys = s[k * 2 + 1];

		x0 = nearbyint(xs);
		y0 = nearbyint(ys);

		for (i = 0; i < 9; i++)
		{
			x = nearbyint(x0 + c_l[i * 2 + 0]);
			y = nearbyint(y0 + c_l[i * 2 + 1]);

			j = y*XDIM + x;

			del = delta(xs, ys, x, y);

			F_s[2 * k + 0] += 2.*(1. * 1. * del)*rho[j] * (u_s[2 * k + 0] - u[2 * j + 0]);
			F_s[2 * k + 1] += 2.*(1. * 1. * del)*rho[j] * (u_s[2 * k + 1] - u[2 * j + 1]);
		}

	}

	__syncthreads();
}

// rho[SIZE]: fluid density	u[2*size]: fluid velocity	f[9*size]: density function		Ns: No. of cilia boundary points	u_s[2*Ns]: cilia velocity	F_s[2*Ns]: cilia force	
// force[2*size]: fluid force	s[2*Ns]: cilia position	XDIM: x dimension	Q: Net flow		epsilon[Ns]: boundary point switching

__global__ void spread(const double * rho, double * u, const double * f, const int Ns, const double * u_s, const double * F_s, double * force, const double * s, const int XDIM, double * Q, const int * epsilon)
{
	int j(0), k(0), x(0), y(0);

	int n(0), m(0);

	double xs(0.), ys(0.), del(0.);

	int size = 192 * XDIM;

	////////////////////////////////////////////////////////////////START//////////////////////////////////////////////////

	const int tile = 128;	//size of a tile, same as blockdim.x

	const int tpoints = tile / 2;

	int numtiles = (2 * Ns - (2 * Ns) % tile) / (tile);	//number of full tiles to populate the whole array of values
	
	int excess = (2 * Ns) % tile;	//number of values outside of full tiles
	
	__shared__ double sh_s[tile];	//shared version of s array
	__shared__ double sh_F_s[tile];	//shared version of F_s array
	__shared__ int sh_epsilon[tile];

	j = blockIdx.x*blockDim.x + threadIdx.x;	//unique thread ID

	n = threadIdx.x;		//thread ID within block

	force[0 * size + j] = 0.;		//initialise
	force[1 * size + j] = 0.;

	sh_s[n] = 0.;
	sh_F_s[n] = 0.;

	x = j%XDIM;
	y = (j - j%XDIM) / XDIM;

	double temp;

	for (m = 0; m < numtiles; m++)		//iterate for each tile within the arrays
	{
		__syncthreads();

		sh_s[n] = s[m*tile + n];		//take values from next tile in the arrays to shared memory
		sh_F_s[n] = F_s[m*tile + n];
		if(n<tpoints) sh_epsilon[n] = epsilon[m*tpoints + n];

		__syncthreads();


		for (k = 0; k < tpoints; k++)	//iterate for each value within a tile ("tile" values reporesent "tile/2" points with x and y coordinates)
		{
			xs = sh_s[2 * k + 0];		//x value
			ys = sh_s[2 * k + 1];		//y value

			del = delta(xs, ys, x, y);

			force[0 * size + j] += sh_F_s[2 * k + 0] * del * 1. * sh_epsilon[k];		//calculate force x
			force[1 * size + j] += sh_F_s[2 * k + 1] * del * 1. * sh_epsilon[k];		//calculate force y

			//__syncthreads();
		}

		__syncthreads();
	}

	

	if (n < excess)		//if there are excess values after the arrays have been split into tiles, and only execute for that many threads
	{
		sh_s[n] = s[numtiles*tile + n];		//take values from excess into shared memory
		sh_F_s[n] = F_s[numtiles*tile + n];
	}
	else
	{
		sh_s[n] = -100.;		//dummy values
		sh_F_s[n] = 0.;
	}
	
	__syncthreads();

		for (k = 0; k < tpoints; k++)	//iterate for all remaining values
		{
			xs = sh_s[k * 2 + 0];		//x value
			ys = sh_s[k * 2 + 1];		//y value

			del = delta(xs, ys, x, y);

			force[0 * size + j] += sh_F_s[2 * k + 0] * del * 1.*epsilon[numtiles*tpoints + k];		//calculate force x
			force[1 * size + j] += sh_F_s[2 * k + 1] * del * 1.*epsilon[numtiles*tpoints + k];		//calculate force y

			//__syncthreads();
		}
		
	__syncthreads();

	//this is the original code, without using shared memory
	/*for (k = 0; k < Ns; k++)
	{
		xs = s[k * 2 + 0];
		ys = s[k * 2 + 1];

		del = delta(xs, ys, x, y);

		force[0 * size + j] += F_s[2 * k + 0] * del * 1.*epsilon[k];
		force[1 * size + j] += F_s[2 * k + 1] * del * 1.*epsilon[k];
	}*/

	/////////////////////////////////////////////////////////////////END////////////////////////////////////////////////////////////

	u[2 * j + 0] = (c_l[0 * 2 + 0] * f[9 * j + 0] + c_l[1 * 2 + 0] * f[9 * j + 1] + c_l[2 * 2 + 0] * f[9 * j + 2] + 
			c_l[3 * 2 + 0] * f[9 * j + 3] + c_l[4 * 2 + 0] * f[9 * j + 4] + c_l[5 * 2 + 0] * f[9 * j + 5] + 
			c_l[6 * 2 + 0] * f[9 * j + 6] + c_l[7 * 2 + 0] * f[9 * j + 7] + c_l[8 * 2 + 0] * f[9 * j + 8] + 0.5*force[0 * size + j]) / rho[j];

	u[2 * j + 1] = (c_l[1 * 2 + 1] * f[9 * j + 1] + c_l[1 * 2 + 1] * f[9 * j + 1] + c_l[2 * 2 + 1] * f[9 * j + 2] +
			c_l[3 * 2 + 1] * f[9 * j + 3] + c_l[4 * 2 + 1] * f[9 * j + 4] + c_l[5 * 2 + 1] * f[9 * j + 5] +
			c_l[6 * 2 + 1] * f[9 * j + 6] + c_l[7 * 2 + 1] * f[9 * j + 7] + c_l[8 * 2 + 1] * f[9 * j + 8] + 0.5*force[1 * size + j]) / rho[j];

	__syncthreads();

	if (x == XDIM - 5)
	{
		//Q[0] += u[2 * j + 0]/192.;
		temp = u[2 * j + 0] / 192.;
		DoubleAtomicAdd(Q, temp);
	}

	__syncthreads();
}


