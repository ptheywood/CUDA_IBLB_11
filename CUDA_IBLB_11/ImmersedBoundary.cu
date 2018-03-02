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

__device__ double delta(const double & xs, const double & ys, const int & x, const int & y)
{
	double deltax(0.), deltay(0.), delta(0.);

	double dx = abs(x - xs);
	double dy = abs(y - ys);

	if (dx <= 1.5)
	{
		if (dx <= 0.5)
		{
			deltax = (1. / 3.)*(1. + sqrt(-3. * dx*dx + 1.));
		}
		else deltax = (1. / 6.)*(5. - 3. * dx - sqrt(-3. * (1. - dx)*(1. - dx) + 1.));
	}

	if (dy <= 1.5)
	{
		if (dy <= 0.5)
		{
			deltay = (1. / 3.)*(1. + sqrt(-3. * dy*dy + 1.));
		}
		else deltay = (1. / 6.)*(5. - 3. * dy - sqrt(-3. * (1. - dy)*(1. - dy) + 1.));
	}

	delta = deltax * deltay;

	return delta;
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

	int size = 200 * XDIM;

	const int tile = 1000;

	int numtiles = (2 * Ns - (2 * Ns) % tile) / (tile);
	
	int overflow = (2 * Ns) % tile;
	
	__shared__ double sh_s[tile];
	__shared__ double sh_F_s[tile];

	j = blockIdx.x*blockDim.x + threadIdx.x;

	n = threadIdx.x;

	force[0 * size + j] = 0.;
	force[1 * size + j] = 0.;

	sh_s[n] = 0.;
	sh_F_s[n] = 0.;

	x = j%XDIM;
	y = (j - j%XDIM) / XDIM;

	for (m = 0; m < numtiles; m++)
	{

		sh_s[n] = 0.;
		sh_F_s[n] = 0.;

		sh_s[n] = s[m*tile + n];
		sh_F_s[n] = F_s[m*tile + n];

		__syncthreads();


		for (k = 0; k < tile/2; k++)
		{
			xs = sh_s[2 * k + 0];
			ys = sh_s[2 * k + 1];

			del = delta(xs, ys, x, y);

			force[0 * size + j] += sh_F_s[2 * k + 0] * del * 1. * epsilon[m*tile/2 + k];
			force[1 * size + j] += sh_F_s[2 * k + 1] * del * 1. * epsilon[m*tile/2 + k];
		}

		__syncthreads();
	}

	if(overflow !=0 && n<overflow)
	{
		sh_s[n] = s[numtiles*tile + n];
		sh_F_s[n] = F_s[numtiles*tile + n];

		__syncthreads();


		for (k = 0; k < (overflow / 2); k++)
		{
			xs = sh_s[k * 2 + 0];
			ys = sh_s[k * 2 + 1];

			del = delta(xs, ys, x, y);

			force[0 * size + j] += sh_F_s[2 * k + 0] * del * 1.*epsilon[m*tile/2 + k];
			force[1 * size + j] += sh_F_s[2 * k + 1] * del * 1.*epsilon[m*tile/2 + k];
		}

		__syncthreads();
	}

	/*for (k = 0; k < Ns; k++)
	{
		xs = s[k * 2 + 0];
		ys = s[k * 2 + 1];

		del = delta(xs, ys, x, y);

		force[0 * size + j] += F_s[2 * k + 0] * del * 1.*epsilon[k];
		force[1 * size + j] += F_s[2 * k + 1] * del * 1.*epsilon[k];
	}*/

	u[2 * j + 0] = (c_l[0 * 2 + 0] * f[9 * j + 0] + c_l[1 * 2 + 0] * f[9 * j + 1] + c_l[2 * 2 + 0] * f[9 * j + 2] + 
			c_l[3 * 2 + 0] * f[9 * j + 3] + c_l[4 * 2 + 0] * f[9 * j + 4] + c_l[5 * 2 + 0] * f[9 * j + 5] + 
			c_l[6 * 2 + 0] * f[9 * j + 6] + c_l[7 * 2 + 0] * f[9 * j + 7] + c_l[8 * 2 + 0] * f[9 * j + 8] + 0.5*force[0 * size + j]) / rho[j];

	u[2 * j + 1] = (c_l[1 * 2 + 1] * f[9 * j + 1] + c_l[1 * 2 + 1] * f[9 * j + 1] + c_l[2 * 2 + 1] * f[9 * j + 2] +
			c_l[3 * 2 + 1] * f[9 * j + 3] + c_l[4 * 2 + 1] * f[9 * j + 4] + c_l[5 * 2 + 1] * f[9 * j + 5] +
			c_l[6 * 2 + 1] * f[9 * j + 6] + c_l[7 * 2 + 1] * f[9 * j + 7] + c_l[8 * 2 + 1] * f[9 * j + 8] + 0.5*force[1 * size + j]) / rho[j];

	if (x == XDIM - 5)
	{
			Q[0] += u[2 * j + 0]/200.;
	}
}


