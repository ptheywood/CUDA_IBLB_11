#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <ctime>
#include <sstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "LatticeBoltzmann.cuh"
#include "ImmersedBoundary.cuh"

#include "seconds.h"



using namespace std;



//------------------------------------------PHYSICAL CONSTANTS----------------------------

#define C_S 0.57735		//SPEED OF SOUND ON LATTICE
#define RHO_0 1.		//FLUID DENSITY
#define PI 3.14159		//PI

//-------------------------------------------PARAMETER SCALING----------------------------

double l_0 = 0.000006;					//6 MICRON CILIUM LENGTH
double t_0 = 0.067;						//67ms BEAT PERIOD AT 15Hz

__constant__ double A_mn[7 * 2 * 3] =						//WITH MUCUS PRESENT
{
	-0.449,	 0.130, -0.169,	 0.063, -0.050, -0.040, -0.068,
	2.076, -0.003,	 0.054,	 0.007,	 0.026,	 0.022,	 0.010,
	-0.072, -1.502,	 0.260, -0.123,	 0.011, -0.009,	 0.196,
	-1.074, -0.230, -0.305, -0.180, -0.069,	 0.001, -0.080,
	0.658,	 0.793, -0.251,	 0.049,	 0.009,	 0.023, -0.111,
	0.381,	 0.331,	 0.193,	 0.082,	 0.029,	 0.002,	 0.048
};

__constant__ double B_mn[7 * 2 * 3] =
{
	0.0, -0.030, -0.093,  0.037,  0.062,  0.016, -0.065,
	0.0,  0.080, -0.044, -0.017,  0.052,  0.007,  0.051,
	0.0,  1.285, -0.036, -0.244, -0.093, -0.137,  0.095,
	0.0, -0.298,  0.513,  0.004, -0.222,  0.035, -0.128,
	0.0, -1.034,  0.050,  0.143,  0.043,  0.098, -0.054,
	0.0,  0.210, -0.367,  0.009,  0.120, -0.024,  0.102
};

//__constant__ double A_mn[7 * 2 * 3] =						//WITHOUT MUCUS
//{
//	-0.654,	 0.393,	-0.097,	 0.079,	 0.119,	 0.119,	 0.009,
//	1.895,	-0.018,	 0.158,	 0.010,	 0.003,	 0.013,	 0.040,
//	0.787,	-1.516,	 0.032,	-0.302,	-0.252,	-0.015,	 0.035,
//	-0.552,	-0.126,	-0.341,	 0.035,	 0.006, -0.029,	-0.068,
//	0.202,	 0.716,	-0.118,	 0.142,	 0.110,	-0.013,	-0.043,
//	0.096,	 0.263,	 0.186,	-0.067,	-0.032,	-0.002,	 0.015
//};
//
//__constant__ double B_mn[7 * 2 * 3] =
//{
//	0.0,	 0.284,	 0.006,	-0.059,	 0.018,	 0.053,	 0.009,
//	0.0,	 0.192,	-0.050,	 0.012,	-0.007,	-0.014,	-0.017,
//	0.0,	 1.045,	 0.317,	 0.226,	 0.004,	-0.082,	-0.040,
//	0.0,	-0.499,	 0.423,	 0.138,	 0.125,	 0.075,	 0.067,
//	0.0,	-1.017,	-0.276,	-0.196,	-0.037,	 0.025,	 0.023,
//	0.0,	 0.339,	-0.327,	-0.114,	-0.105,	-0.057,	-0.055
//};


__global__ void define_filament(const int T, const int it, const double c_space, const int p_step, const int c_num, const int c_rows, float * s, float * lasts, float * b_points, int length)
{
	int n(0), j(0);

	//int length = 128;

	int f_length = length * 100;

	int l_max = int(length * 1.16);

	float arcl(0.);
	int phase(0);
	int z_phase(0);

	float b_length(0.);

	float a_n[2 * 7];
	float b_n[2 * 7];

	int threadnum = blockDim.x*blockIdx.x + threadIdx.x;		//individual thread number

	int k = threadnum % f_length;								//point number along cilium

	int m = ((threadnum - k) / f_length) % c_num;				//cilium number within row

	int o = (threadnum - k) / f_length;							//overall cilium number

	int r = (o - m) / c_num;									//row number

	int point = threadnum;

	
		arcl = 1.*k / f_length;

		if (it + m*p_step == T) phase = T;
		else phase = (it + m*p_step + r*p_step) % T;//(it + m*p_step) % T;

		float offset = 1.*(m - (c_num - 1) / 2.)*c_space;

		for (n = 0; n < 7; n++)
		{
			a_n[2 * n + 0] = 0.;
			b_n[2 * n + 0] = 0.;

			a_n[2 * n + 0] += A_mn[n + 14 * 0 + 7 * 0] * pow(arcl, 0 + 1);
			b_n[2 * n + 0] += B_mn[n + 14 * 0 + 7 * 0] * pow(arcl, 0 + 1);

			a_n[2 * n + 0] += A_mn[n + 14 * 1 + 7 * 0] * pow(arcl, 1 + 1);
			b_n[2 * n + 0] += B_mn[n + 14 * 1 + 7 * 0] * pow(arcl, 1 + 1);

			a_n[2 * n + 0] += A_mn[n + 14 * 2 + 7 * 0] * pow(arcl, 2 + 1);
			b_n[2 * n + 0] += B_mn[n + 14 * 2 + 7 * 0] * pow(arcl, 2 + 1);

			a_n[2 * n + 1] = 0.;
			b_n[2 * n + 1] = 0.;

			a_n[2 * n + 1] += A_mn[n + 14 * 0 + 7 * 1] * pow(arcl, 0 + 1);
			b_n[2 * n + 1] += B_mn[n + 14 * 0 + 7 * 1] * pow(arcl, 0 + 1);

			a_n[2 * n + 1] += A_mn[n + 14 * 1 + 7 * 1] * pow(arcl, 1 + 1);
			b_n[2 * n + 1] += B_mn[n + 14 * 1 + 7 * 1] * pow(arcl, 1 + 1);

			a_n[2 * n + 1] += A_mn[n + 14 * 2 + 7 * 1] * pow(arcl, 2 + 1);
			b_n[2 * n + 1] += B_mn[n + 14 * 2 + 7 * 1] * pow(arcl, 2 + 1);

		}

		s[7 * point + 0] = 1. * l_max * a_n[2 * 0 + 0] * 0.5 + offset;	//x position
		s[7 * point + 1] = 1. * l_max * a_n[2 * 0 + 1] * 0.5;			//y position
		s[7 * point + 2] = c_space*(0.5 + r);							//z position
		s[7 * point + 3] = l_max * arcl;								//position on cilium

		for (n = 1; n < 7; n++)
		{
			s[7 * point + 0] += 1. * l_max * (a_n[2 * n + 0] * cos(n*2.*PI*phase / T) + b_n[2 * n + 0] * sin(n*2.*PI*phase / T));
			s[7 * point + 1] += 1. * l_max * (a_n[2 * n + 1] * cos(n*2.*PI*phase / T) + b_n[2 * n + 1] * sin(n*2.*PI*phase / T));
		}

		if (it > 0)
		{
			s[7 * point + 4] = s[7 * point + 0] - lasts[3 * point + 0];
			s[7 * point + 5] = s[7 * point + 1] - lasts[3 * point + 1];
			s[7 * point + 6] = 0.;
		}
		

		lasts[3 * point + 0] = s[7 * point + 0];
		lasts[3 * point + 1] = s[7 * point + 1];
		lasts[3 * point + 2] = s[7 * point + 2];
	

	for (j = o*length ; j < (o + 1)*length; j++)
	{
		b_length = j%length;

		if (abs(s[7 * point + 3] - b_length) < 0.01)
		{
			b_points[7 * j + 0] = s[7 * point + 0];
			b_points[7 * j + 1] = s[7 * point + 1];
			b_points[7 * j + 2] = s[7 * point + 2];

			b_points[7 * j + 3] = s[7 * point + 4];
			b_points[7 * j + 4] = s[7 * point + 5];
			b_points[7 * j + 5] = s[7 * point + 6];

		}
		
	}
}

__global__ void boundary_check(const double c_space, const int c_num, const int XDIM, const int it, const float *  b_points,  float * s, float * u_s, int * epsilon, int length)
{
	int r(0), j(0), l(0), m(0);

	//int length = 128;

	bool xclose = 0;
	bool yclose = 0;

	int r_max = 2 * length / c_space;

	float x_m(0.), y_m(0.), z_m(0.), x_l(0.), y_l(0.), z_l(0.);

	j = blockIdx.x*blockDim.x + threadIdx.x;

	
	{
		s[3 * j + 0] = (c_space*c_num) / 2. + b_points[7 * j + 0];

		if (s[3 * j + 0] < 0) s[3 * j + 0] += XDIM;
		else if (s[3 * j + 0] > XDIM) s[3 * j + 0] -= XDIM;

		s[3 * j + 1] = b_points[7 * j + 1] + 1;
		s[3 * j + 2] = b_points[7 * j + 2];

		if (it == 0)
		{
			u_s[3 * j + 0] = 0.;
			u_s[3 * j + 1] = 0.;
			u_s[3 * j + 2] = 0.;
		}
		else
		{
			u_s[3 * j + 0] = b_points[7 * j + 3];
			u_s[3 * j + 1] = b_points[7 * j + 4];
			u_s[3 * j + 2] = b_points[7 * j + 5];
		}

		epsilon[j] = 1;
	}

	__syncthreads();

	
	{
			m = (j - j%length) / length;

			x_m = s[3 * j + 0];
			y_m = s[3 * j + 1];
			z_m = s[3 * j + 2];

			for (r = 1; r < r_max; r++)
			{
				for (l = 0; l < length; l++)
				{
					xclose = 0;
					yclose = 0;

					if (m - r < 0)
					{
						x_l = s[3 * (l + (m - r + c_num) * length) + 0];
						y_l = s[3 * (l + (m - r + c_num) * length) + 1];
						z_l = s[3 * (l + (m - r + c_num) * length) + 2];
					}
					else
					{
						x_l = s[3 * (l + (m - r) * length) + 0];
						y_l = s[3 * (l + (m - r) * length) + 1];
						z_l = s[3 * (l + (m - r) * length) + 2];
					}

					if (abs(x_l - x_m) < 1 && z_l == z_m) xclose = 1;

					if (abs(y_l - y_m) < 1 && z_l == z_m) yclose = 1;

					if (xclose && yclose) epsilon[j] = 0;

				}
	}
}


}


template <typename T>
std::string to_string_3(const T a_value, const int n = 3)
{
	std::ostringstream out;
	out << std::setprecision(n) << a_value;
	return out.str();
}

int main(int argc, char * argv[])
{
	//----------------------------INITIALISING----------------------------

	unsigned int c_fraction = 1;
	unsigned int c_num = 12;
	unsigned int c_rows = 1;			//ROWS IF CILIA IN SIMULATION
	double Re = 1.0;
	unsigned int XDIM = 192;
	unsigned int YDIM = 192;
	unsigned int ZDIM =  16; 
	unsigned int T = 1000000;
	unsigned int T_pow = 1;
	float T_num = 1.0;
	unsigned int ITERATIONS = T;
	unsigned int P_num = 100;
	float I_pow = 1.0;
	unsigned int INTERVAL = 500;
	unsigned int LENGTH = 96;
	unsigned int c_space = 16;
	int GPU = 0;
	bool BigData = 0;
	float G_PM = 6.; //6.

	if (argc < 12)
	{
		cout << "Too few arguments! " << argc - 1 << " entered of 11 required. " << endl;

		return 1;
	}
	
	stringstream arg;

	arg << argv[1] << ' ' << argv[2] << ' ' << argv[3] << ' ' << argv[4] << ' ' << argv[5] 
		<< ' ' << argv[6] << ' ' << argv[7] << ' ' << argv[8] << ' ' << argv[9] << ' ' << argv[10] << ' ' << argv[11];

	arg >> c_fraction >> c_num >> c_rows >> c_space >> Re >> T_num >> T_pow >> I_pow >> P_num >> GPU >> BigData;

	XDIM = c_num*c_space;
	ZDIM = c_space*c_rows;
	T = nearbyint(T_num * pow(10, T_pow));
	ITERATIONS = T*I_pow; 
	//cout << endl << T << " / " << I_pow << " = " << ITERATIONS << " iterations" << endl;
	INTERVAL = ITERATIONS / P_num;
	//cout << endl << ITERATIONS << " / " << P_num << " = " << INTERVAL << " interval (should be 1)" << endl;

	if (XDIM < 2 * LENGTH && c_num > 1)
	{
		cout << "not enough cilia in simulation! cilia spacing of " << c_space << " requires at least " << 2 * LENGTH / c_space << " cilia" << endl;

		//return 1;
	}

	double dx = 1. / LENGTH;
	double dt = 1. / T;
	double SPEED = 1000./T; 

	double t_scale = 1000.*dt*t_0;					//milliseconds
	double x_scale = 1000000. * dx*l_0;				//microns
	double s_scale = x_scale / t_scale;				//millimetres per second 

	const double TAU_P = (SPEED*LENGTH) / (Re*C_S*C_S) + 1. / 2.;
	const double TAU_M = 5. * (SPEED*LENGTH) / (Re*C_S*C_S) + 1. / 2.;
	//const double TAU2 = 1. / (4.*(TAU - (1. / 2.))) + 1. / 2.;    //lamda = 1/4 most stable, 1/12 for optimal results

	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);

	cout << asctime(timeinfo) << endl;

	cout << "Initialising...\n";

	unsigned int i(0), j(0), k(0);

	unsigned int it(0);
	//int phase(0);
	int p_step = T * c_fraction / c_num;

	float * lasts;
	lasts = new float[3 * c_num * c_rows * LENGTH * 100];

	float * boundary;
	boundary = new float[7 * c_num * c_rows * LENGTH * 100];

	int Np = LENGTH * c_num * c_rows;
	
	const int size = XDIM*YDIM*ZDIM;

	for (k = 0; k < c_num * c_rows * LENGTH * 100; k++)
	{
		boundary[7 * k + 0] = 0.;
		boundary[7 * k + 1] = 0.;
		boundary[7 * k + 2] = 0.;
		boundary[7 * k + 3] = 0.;
		boundary[7 * k + 4] = 0.;
		boundary[7 * k + 5] = 0.;
		boundary[7 * k + 6] = 0.;

		lasts[3 * k + 0] = 0.;
		lasts[3 * k + 1] = 0.;
		lasts[3 * k + 2] = 0.;

	}

	

	//-------------------------------CUDA PARAMETERS DEFINITION-----------------------


	int blocksize = 128;											//for lattice boltzmann kernels

	int gridsize = size / blocksize;

	int blocksize2 = c_num * c_rows * LENGTH;							//for spread kernel

	int gridsize2 = 1;

	if (blocksize2 > 1024)
	{
		for (blocksize2 = 1024; blocksize2 > 0; blocksize2 -= LENGTH)
		{
			if ((c_num*LENGTH*c_rows) % blocksize2 == 0)
			{
				gridsize2 = (c_num*LENGTH*c_rows) / blocksize2;
				break;
			}
		}
	}

	int blocksize3 = 96;											//for define_filament kernel
	int gridsize3 = c_num * c_rows * LENGTH * 100 / blocksize3;

	int blocksize4 = blocksize;

	int gridsize4 = (15 * LENGTH * c_num * c_rows) / blocksize4;

	cudaError_t cudaStatus;

	double * Q;
	cudaMallocHost(&Q, sizeof(double));
	Q[0] = 0.;

	double * Q_M;
	cudaMallocHost(&Q_M, sizeof(double));
	Q_M[0] = 0.;

	double * Q_P;
	cudaMallocHost(&Q_P, sizeof(double));
	Q_P[0] = 0.;

	int * nodes;
	nodes = new int[15 * LENGTH * c_num * c_rows];



	cudaStatus = cudaSetDevice(GPU);
	

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Failed to set CUDA device.\n");
	}

	//------------------------------------------ERROR------------------------------------------------


	//double l_error = (l_0*dx)*(l_0*dx);
	//double t_error = (t_0*dt)*(t_0*dt);
	//double c_error = (t_0*dt)*(t_0*dt) / ((l_0*dx)*(l_0*dx));
	double Ma = 1.*SPEED / C_S;
	time_t p_runtime;


	//-------------------------------------------ASSIGN CELL VALUES ON HEAP-----------------------------

	double * u;								//VELOCITY VECTOR

	u = new double[3 * size];

	double * rho;							//TOTAL DENSITY

	rho = new double[size];

	double * rho_P;							//PCL DENSITY

	rho_P = new double[size];

	double * rho_M;							//MUCUS DENSITY

	rho_M = new double[size];

	double * f0_P;							//PCL EQUILIBRIUM DISTRIBUTION FUNCTION

	f0_P = new double[15 * size];

	double * f0_M;							//MUCUS EQUILIBRIUM DISTRIBUTION FUNCTION

	f0_M = new double[15 * size];

	double * f_P;							//PCL DISTRIBUTION FUNCTION

	f_P = new double[15 * size];

	double * f_M;							//MUCUS DISTRIBUTION FUNCTION

	f_M = new double[15 * size];

	double * f1_P;							//PCL POST COLLISION DISTRIBUTION FUNCTION

	f1_P = new double[15 * size];

	double * f1_M;							//MUCUS POST COLLISION DISTRIBUTION FUNCTION

	f1_M = new double[15 * size];

	double * force;							//MACROSCOPIC BODY FORCE VECTOR

	force = new double[3 * size];

	double * force_P;						//PCL MACROSCOPIC FORCE

	force_P = new double[3 * size];

	double * force_M;						//MUCUS MACROSCOPIC FORCE

	force_M = new double[3 * size];

	double * u_M;						//MUCUS VELOCITY

	u_M = new double[3 * size];

	double * force_E;						//ELASTIC MACROSCOPIC FORCE

	force_E = new double[3 * size];

	double * F_P;							//PCL LATTICE BOLTZMANN FORCE

	F_P = new double[15 * size];

	double * F_M;							//MUCUS LATTICE BOLTZMANN FORCE

	F_M = new double[15 * size];

	double * F_E;							//ELASTIC LATTICE BOLTZMANN FORCE

	F_E = new double[15 * size];

	unsigned int Ns = LENGTH * c_num * c_rows;		//NUMBER OF BOUNDARY POINTS


	float * s;							//BOUNDARY POINTS

	float * u_s;						//BOUNDARY POINT VELOCITY

	float * F_s;						//BOUNDARY FORCE

	int * epsilon;

	s = new float[3 * Ns];

	u_s = new float[3 * Ns];

	F_s = new float[2 * Ns];

	epsilon = new int[Ns];

	for (k = 0; k < Ns; k++)
	{
		epsilon[k] = 1;
	}

	for (k = 0; k < 15 * LENGTH * c_num * c_rows; k++)
	{
		nodes[k] = 0;
	}


	//----------------------------------------CREATE DEVICE VARIABLES-----------------------------

	double * d_u;							//VELOCITY VECTOR

	double * d_rho;							//DENSITY

	double * d_rho_P;						//PCL DENSITY

	double * d_rho_M;						//MUCUS DENSITY

	double * d_f0_P;						//PCL EQUILIBRIUM DISTRIBUTION FUNCTION

	double * d_f0_M;						//MUCUS EQUILIBRIUM DISTRIBUTION FUNCTION

	double * d_f_P;							//PCL DISTRIBUTION FUNCTION

	double * d_f_M;							//MUCUS DISTRIBUTION FUNCTION

	double * d_f1_P;						//POST COLLISION DISTRIBUTION FUNCTION

	double * d_f1_M;						//POST COLLISION DISTRIBUTION FUNCTION

	//double * d_centre;

	double * d_force;

	double * d_force_P;

	double * d_force_M;

	double * d_u_M;

	double * d_force_E;

	double * d_F_P;

	double * d_F_M;

	double * d_F_E;

	float * d_F_s;

	float * d_s;

	float * d_u_s;

	int * d_epsilon;

	double * d_Q;

	double * d_Q_M;

	double * d_Q_P;

	int * d_nodes;

	float * d_lasts;

	float * d_boundary;

	float * d_b_points;



	//---------------------------CUDA MALLOC-------------------------------------------------------------
	{
		cudaStatus = cudaMalloc((void**)&d_u, 3 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_rho, size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of rho failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_rho_P, size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of rho_P failed!");
		}
		
		cudaStatus = cudaMalloc((void**)&d_f0_P, 15 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of f0_P failed\n");
		}
		
		cudaStatus = cudaMalloc((void**)&d_f_P, 15 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of f_P failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_f1_P, 15 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of f1_P failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_force, 3 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of force failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_force_P, 3 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of force_P failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_F_P, 15 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc F_P failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_Q, sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_Q_P, sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		
			cudaStatus = cudaMalloc((void**)&d_rho_M, size * sizeof(double));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc of rho_M failed\n");
			}

			cudaStatus = cudaMalloc((void**)&d_f0_M, 15 * size * sizeof(double));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc of f0_M failed\n");
			}

			cudaStatus = cudaMalloc((void**)&d_f_M, 15 * size * sizeof(double));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc of f_M failed\n");
			}

			cudaStatus = cudaMalloc((void**)&d_f1_M, 15 * size * sizeof(double));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc of f1_M failed\n");
			}

			cudaStatus = cudaMalloc((void**)&d_force_M, 3 * size * sizeof(double));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc of force_M failed\n");
			}

			cudaStatus = cudaMalloc((void**)&d_F_M, 15 * size * sizeof(double));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc F_M failed\n");
			}

			cudaStatus = cudaMalloc((void**)&d_Q_M, sizeof(double));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
			}

		cudaStatus = cudaMalloc((void**)&d_force_E, 3 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of force_E failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_F_E, 15 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc F_E failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_u_M, 3 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc F_E failed\n");
		}
		

	

		cudaStatus = cudaMalloc((void**)&d_F_s, 2 * Ns * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of F_s failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_s, 3 * Ns * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of s failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_u_s, 3 * Ns * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u_s failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_epsilon, Ns * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of epsilon failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_lasts, 3 * c_num * c_rows * LENGTH * 100 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u_s failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_boundary, 7 * c_num * c_rows * LENGTH * 100 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u_s failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_b_points, 7 * Np * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u_s failed\n");
		}

		cudaStatus = cudaMalloc((void**)&d_nodes, 15 * LENGTH * c_num * c_rows * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of nodes failed\n");
		}
	}

	//----------------------------------------DEFINE DIRECTORIES----------------------------------
	
	string output_data = "Data/Test/";

	if(ShARC) output_data = "/shared/soft_matter_physics2/User/Phq16ja/ShARC_Data/";
	else output_data = "C:/Users/phq16ja/Documents/Data/";
		//output_data = "//uosfstore.shef.ac.uk/shared/soft_matter_physics2/User/Phq16ja/Local_Data/";

	string raw_data = output_data + "Raw/";
	raw_data += to_string(c_num);
	raw_data += "/";
	raw_data += to_string(c_fraction);
	raw_data += "/";

	string cilia_data =  output_data + "Cilia/";
	cilia_data += to_string(c_num);
	cilia_data += "/";
	cilia_data += to_string(c_fraction);
	cilia_data += "/";

	string img_data = output_data + "Img/";
	img_data += to_string(c_num);
	img_data += "/";
	

	string outfile = cilia_data;

	//----------------------------------------BOUNDARY INITIALISATION------------------------------------------------

	string flux = output_data + "/Flux/" + to_string(c_fraction) + "_" + to_string(c_num) + "_" + to_string(c_space) + "_" + to_string_3(Re) + "_" + to_string_3(T_num) + "x" + to_string_3(T_pow) + "-flux.dat";

	string parameters = raw_data + "/SimLog.txt";


	ofstream fsA(output_data.c_str());

	ofstream fsB(flux.c_str());

	ofstream fsC(parameters.c_str());

	ofstream fsD;

	fsB.open(flux.c_str(), ofstream::trunc);

	fsB.close();

	fsC.open(parameters.c_str(), ofstream::trunc);

	fsC.close();



	//----------------------------------------INITIALISE ALL CELL VALUES---------------------------------------

	for (j = 0; j < XDIM*YDIM*ZDIM; j++)
	{
		rho[j] = RHO_0;

		int y = ((j - (j % XDIM)) / XDIM) % YDIM;

		if (y < LENGTH*0.9) //LENGTH*0.9
			{
				rho_P[j] = 0.95; //0.95
				rho_M[j] = 0.05; //0.05
			}

		if (y >= LENGTH*0.9)
			{
				rho_P[j] = 0.05; //0.05
				rho_M[j] = 0.95; //0.95
			}
		
		

		u[0 * size + j] = 0.0;
		u[1 * size + j] = 0.0;
		u[2 * size + j] = 0.0;

		force[0 * size + j] = 0.;
		force[1 * size + j] = 0.;
		force[2 * size + j] = 0.;

		force_P[0 * size + j] = 0.;
		force_P[1 * size + j] = 0.;
		force_P[2 * size + j] = 0.;

		force_M[0 * size + j] = 0.;
		force_M[1 * size + j] = 0.;
		force_M[2 * size + j] = 0.;

		u_M[0 * size + j] = 0.;
		u_M[1 * size + j] = 0.;
		u_M[2 * size + j] = 0.;

		force_E[0 * size + j] = 0.;
		force_E[1 * size + j] = 0.;
		force_E[2 * size + j] = 0.;


		for (i = 0; i < 15; i++)
		{
			f0_P[15 * j + i] = 0.;
			f_P[15 * j + i] = 0.;
			f1_P[15 * j + i] = 0.;
			F_P[15 * j + i] = 0.;

			f0_M[15 * j + i] = 0.;
			f_M[15 * j + i] = 0.;
			f1_M[15 * j + i] = 0.;
			F_M[15 * j + i] = 0.;

			F_E[15 * j + i] = 0.;
		}

	}

	//------------------------------------------------------COPY INITIAL VALUES TO DEVICE-----------------------------------------------------------

	//CUDA MEMORY COPIES
	{
		cudaStatus = cudaMemcpy(d_u, u, 3 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed\n");
		}

		cudaStatus = cudaMemcpy(d_rho, rho, size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of rho failed\n");
		}

		cudaStatus = cudaMemcpy(d_rho_P, rho_P, size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of rho_P failed\n");
		}
		cudaStatus = cudaMemcpy(d_rho_M, rho_M, size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of rho_M failed\n");
		}

		cudaStatus = cudaMemcpy(d_f0_P, f0_P, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of f0_P failed\n");
		}

		cudaStatus = cudaMemcpy(d_f0_M, f0_M, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of f0_M failed\n");
		}

		cudaStatus = cudaMemcpy(d_f_P, f_P, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of f_P failed\n");
		}

		cudaStatus = cudaMemcpy(d_f_M, f_M, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of f_M failed\n");
		}

		cudaStatus = cudaMemcpy(d_f1_P, f1_P, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of f1_P failed\n");
		}

		cudaStatus = cudaMemcpy(d_f1_M, f1_M, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of f1_M failed\n");
		}

		cudaStatus = cudaMemcpy(d_force, force, 3 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of force failed\n");
		}

		cudaStatus = cudaMemcpy(d_force_P, force_P, 3 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of force_P failed\n");
		}

		cudaStatus = cudaMemcpy(d_force_M, force_M, 3 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of force_M failed\n");
		}

		cudaStatus = cudaMemcpy(d_u_M, u_M, 3 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of force_M failed\n");
		}

		cudaStatus = cudaMemcpy(d_force_E, force_E, 3 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of force_M failed\n");
		}

		cudaStatus = cudaMemcpy(d_F_P, F_P, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of F_P failed\n");
		}

		cudaStatus = cudaMemcpy(d_F_M, F_M, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of F_M failed\n");
		}

		cudaStatus = cudaMemcpy(d_F_E, F_E, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of F_M failed\n");
		}

		cudaStatus = cudaMemcpy(d_F_s, F_s, 2 * Ns * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of F_s failed!\n");
		}

		cudaStatus = cudaMemcpy(d_lasts, lasts, 3 * Ns * 100 * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of lasts failed!\n"); }

		cudaStatus = cudaMemcpy(d_boundary, boundary, 7 * Ns * 100 * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of boundary failed!\n"); }


		cudaStatus = cudaMemcpy(d_Q, Q, sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of Q failed!\n");
		}

		cudaStatus = cudaMemcpy(d_Q_M, Q_M, sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of Q failed!\n");
		}

		cudaStatus = cudaMemcpy(d_Q_P, Q_P, sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of Q failed!\n");
		}

		cudaStatus = cudaMemcpy(d_epsilon, epsilon, Ns*sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of Q failed!\n");
		}

		cudaStatus = cudaMemcpy(d_nodes, nodes, 15 * LENGTH * c_num * c_rows * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of nodes failed!\n");
		}


	}

	//------------------------------------------------------SET INITIAL DISTRIBUTION TO EQUILIBRIUM-------------------------------------------------

	equilibrium << <gridsize, blocksize >> > (d_u, d_rho_P, d_f0_P, d_force_P, d_F_P, XDIM, YDIM, ZDIM, TAU_P);				//PCL INITIAL EQUILIBRIUM SET

	equilibrium << <gridsize, blocksize >> > (d_u, d_rho_M, d_f0_M, d_force_M, d_F_M, XDIM, YDIM, ZDIM, TAU_M);				//ML INITIAL EQUILIBRIUM SET

	{																										// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "first equilibrium launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaMemcpy(f0_P, d_f0_P, 15 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(f0_M, d_f0_M, 15 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
/*
		cudaStatus = cudaMemcpy(F, d_F, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}
*/

	}

	srand(3);

	for (j = 0; j < XDIM*YDIM*ZDIM; j++)
	{
		for (i = 0; i < 15; i++)
		{
			f_P[15 * j + i] = f0_P[15 * j + i];

			f_M[15 * j + i] = f0_M[15 * j + i];
		}

		//initially random fluid motion

		/*f_M[9 * j + 0] = 7. / 8.;
		f_M[9 * j + 1] = 1. / 32. * rand() / RAND_MAX;
		f_M[9 * j + 2] = 1. / 32. * rand() / RAND_MAX;
		f_M[9 * j + 3] = 1. / 32. - f_M[9 * j + 1];
		f_M[9 * j + 4] = 1. / 32. - f_M[9 * j + 2];
		f_M[9 * j + 5] = 1. / 32. * rand() / RAND_MAX;
		f_M[9 * j + 6] = 1. / 32. * rand() / RAND_MAX;
		f_M[9 * j + 7] = 1. / 32. - f_M[9 * j + 5];
		f_M[9 * j + 8] = 1. / 32. - f_M[9 * j + 6];*/


		
	}

	cudaStatus = cudaMemcpy(d_f_P, f_P, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "second cudaMemcpy of f_P failed\n");
	}

	
	{
		cudaStatus = cudaMemcpy(d_f_M, f_M, 15 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "second cudaMemcpy of f_M failed\n");
		}
	}


	//-----------------------------------------------------OUTPUT PARAMETERS------------------------------------------------------------------------


	fsC.open(parameters.c_str(), ofstream::trunc);

	fsC.close();

	fsC.open(parameters.c_str(), ofstream::app);

	fsC << asctime(timeinfo) << endl;
	fsC << "Size: " << XDIM << "x" << YDIM << endl;
	fsC << "Iterations: " << ITERATIONS << endl;
	fsC << "Reynolds Number: " << Re << endl;
	fsC << "Relaxation time: " << TAU_P << ", " << TAU_M <</* ", " << TAU2 <<*/ endl;
	//if (TAU <= 0.6) fsC << "POSSIBLE INSTABILITY! 
	//cout << "\nRelaxation time: " << TAU << endl;
	//if (TAU >= 2.01) fsC << "POSSIBLE INACCURACY! Relaxation time: " << TAU << endl;

	fsC << "Spatial step: " << dx*l_0 << "m" << endl;
	fsC << "Time step: " << dt*t_0 << "s" << endl;
	fsC << "Mach number: " << Ma << endl;
	//fsC << "Spatial discretisation error: " << l_error << endl;
	//fsC << "Time discretisation error: " << t_error << endl;
	//fsC << "Compressibility error: " << c_error << endl;
	fsC << "Phase Step: " << c_fraction << "/" << c_num << endl;

	//fsC << "\nThreads per block: " << blocksize << endl;
	//fsC << "Blocks: " << gridsize << endl;

	if (BigData) fsC << "\nBig Data is ON" << endl;
	else fsC << "\nBig Data is OFF" << endl;

	if (ShARC) fsC << "Running on ShARC" << endl;
	else fsC << "Running on local GPU" << endl;


	cudaStream_t c_stream;
	cudaStream_t f_stream;
	cudaStream_t o_stream;

	cudaStreamCreate(&c_stream);
	cudaStreamCreate(&f_stream);
	cudaStreamCreate(&o_stream);

	cudaEvent_t cilia_done;
	cudaEvent_t fluid_done;
	cudaEvent_t Q_done;

	
	cudaEventCreate(&fluid_done);
	cudaEventRecord(fluid_done, f_stream);
	cudaEventCreate(&Q_done);
	cudaEventRecord(Q_done, f_stream);
	

	//--------------------------ITERATION LOOP-----------------------------
	cout << "Running Simulation...\n";

	time_t start = seconds();

	for (it = 0; it < ITERATIONS; it++)
	{

		cudaStatus = cudaMemcpy(d_force, force, 3 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of force failed\n");
		}

		//--------------------------CILIA BEAT DEFINITION-------------------------

		cudaEventCreate(&cilia_done);

		define_filament << <gridsize3, blocksize3, 0, c_stream >> > (T, it, c_space, p_step, c_num, c_rows, d_boundary, d_lasts, d_b_points, LENGTH);

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) { fprintf(stderr, "define_filament failed: %s\n", cudaGetErrorString(cudaStatus)); }
		}

		cudaStreamWaitEvent(c_stream, fluid_done, 0);
		cudaEventDestroy(fluid_done);

		boundary_check << <gridsize2, blocksize2, 0, c_stream >> > (c_space, c_num, XDIM, it, d_b_points, d_s, d_u_s, d_epsilon, LENGTH);

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) { fprintf(stderr, "boundary_check failed: %s\n", cudaGetErrorString(cudaStatus)); }
		}

		cudaEventRecord(cilia_done, c_stream);

		
			//---------------------------IMMERSED BOUNDARY LATTICE BOLTZMANN STEPS-------------------

		cudaEventCreate(&fluid_done);

		cudaStreamWaitEvent(f_stream, Q_done, 0);
		cudaEventDestroy(Q_done);

		

		equilibrium << <gridsize, blocksize, 0, f_stream >> > (d_u, d_rho_P, d_f0_P, d_force_P, d_F_P, XDIM, YDIM, ZDIM, TAU_P);					//PCL EQUILIBRIUM STEP

		{																										// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "PCL equilibrium launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		equilibrium << <gridsize, blocksize, 0, f_stream >> > (d_u, d_rho_M, d_f0_M, d_force_M, d_F_M, XDIM, YDIM, ZDIM, TAU_M);					//MUCUS EQUILIBRIUM STEP

		{																										// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "mucus equilibrium launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		//////////////////////////////////////////////////////
/*
		cudaStatus = cudaMemcpy(f0_M, d_f0_M, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		cudaStatus = cudaMemcpy(f0_P, d_f0_P, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		cudaStatus = cudaMemcpy(f_M, d_f_M, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		cudaStatus = cudaMemcpy(f_P, d_f_P, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		cudaStatus = cudaMemcpy(F_M, d_F_M, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		cudaStatus = cudaMemcpy(F_P, d_F_P, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		outfile = raw_data + to_string(it) + "-fluid.dat";

		fsA.open(outfile.c_str());

		fsA << "equilibrium" << endl;


		for (j = 63 * 128 + 0; j < 64 * 128 + 128; j++)
		{
			if (j == 63 * 128 + 64 || j == 64 * 128 + 64)
			{
				int x = j%XDIM;
				int y = (j - j%XDIM) / XDIM;

				fsA << x << "\t" << y << "\t" << f0_M[9 * j + 0] << "\t" << f0_M[9 * j + 1] << "\t" << f0_M[9 * j + 2] << "\t" << f0_M[9 * j + 3] << "\t" << f0_M[9 * j + 4]
					<< "\t" << f0_M[9 * j + 5] << "\t" << f0_M[9 * j + 6] << "\t" << f0_M[9 * j + 7] << "\t" << f0_M[9 * j + 8] << endl;

				fsA << x << "\t" << y << "\t" << f0_P[9 * j + 0] << "\t" << f0_P[9 * j + 1] << "\t" << f0_P[9 * j + 2] << "\t" << f0_P[9 * j + 3] << "\t" << f0_P[9 * j + 4]
					<< "\t" << f0_P[9 * j + 5] << "\t" << f0_P[9 * j + 6] << "\t" << f0_P[9 * j + 7] << "\t" << f0_P[9 * j + 8] << "\t" << endl;
			}
		}

		fsA << endl;

		fsA << "distribution" << endl;


		for (j = 63 * 128 + 0; j < 64 * 128 + 128; j++)
		{
			if (j == 63 * 128 + 64 || j == 64 * 128 + 64)
			{
				int x = j%XDIM;
				int y = (j - j%XDIM) / XDIM;

				fsA << x << "\t" << y << "\t" << f_M[9 * j + 0] << "\t" << f_M[9 * j + 1] << "\t" << f_M[9 * j + 2] << "\t" << f_M[9 * j + 3] << "\t" << f_M[9 * j + 4]
					<< "\t" << f_M[9 * j + 5] << "\t" << f_M[9 * j + 6] << "\t" << f_M[9 * j + 7] << "\t" << f_M[9 * j + 8] << endl;

				fsA << x << "\t" << y << "\t" << f_P[9 * j + 0] << "\t" << f_P[9 * j + 1] << "\t" << f_P[9 * j + 2] << "\t" << f_P[9 * j + 3] << "\t" << f_P[9 * j + 4]
					<< "\t" << f_P[9 * j + 5] << "\t" << f_P[9 * j + 6] << "\t" << f_P[9 * j + 7] << "\t" << f_P[9 * j + 8] << "\t" << endl;
			}
		}

		fsA << endl;

		fsA << "force" << endl;


		for (j = 63 * 128 + 0; j < 64 * 128 + 128; j++)
		{
			if (j == 63 * 128 + 64 || j == 64 * 128 + 64)
			{
				int x = j%XDIM;
				int y = (j - j%XDIM) / XDIM;

				fsA << x << "\t" << y << "\t" << F_M[9 * j + 0] << "\t" << F_M[9 * j + 1] << "\t" << F_M[9 * j + 2] << "\t" << F_M[9 * j + 3] << "\t" << F_M[9 * j + 4]
					<< "\t" << F_M[9 * j + 5] << "\t" << F_M[9 * j + 6] << "\t" << F_M[9 * j + 7] << "\t" << F_M[9 * j + 8] << endl;

				fsA << x << "\t" << y << "\t" << F_P[9 * j + 0] << "\t" << F_P[9 * j + 1] << "\t" << F_P[9 * j + 2] << "\t" << F_P[9 * j + 3] << "\t" << F_P[9 * j + 4]
					<< "\t" << F_P[9 * j + 5] << "\t" << F_P[9 * j + 6] << "\t" << F_P[9 * j + 7] << "\t" << F_P[9 * j + 8] << "\t" << endl;
			}
		}
		
		fsA << endl;

		fsA.close();
*/
		//////////////////////////////////////////////

		

		collision << <gridsize, blocksize, 0, f_stream >> > (d_f0_P, d_f_P, d_f1_P, d_F_P, TAU_P, XDIM, YDIM, ZDIM);					//PCL COLLISION STEP

		{																										// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "PCL collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		collision << <gridsize, blocksize, 0, f_stream >> > (d_f0_M, d_f_M, d_f1_M, d_F_M, TAU_M, XDIM, YDIM, ZDIM);					// MUCUS COLLISION STEP

		{																										// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "mucus collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		//////////////////////////////////////////////////////
/*
		cudaStatus = cudaMemcpy(f1_M, d_f1_M, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		cudaStatus = cudaMemcpy(f1_P, d_f1_P, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		outfile = raw_data + to_string(it) + "-fluid.dat";

		fsA.open(outfile.c_str(), ofstream::app);

		fsA << "collision" << endl;


		for (j = 63*128 + 0; j < 64*128 + 128; j++)
		{
			if (j == 63 * 128 + 64 || j == 64 * 128 + 64)
			{
				int x = j%XDIM;
				int y = (j - j%XDIM) / XDIM;

				fsA << x << "\t" << y << "\t" << f1_M[9 * j + 0] << "\t" << f1_M[9 * j + 1] << "\t" << f1_M[9 * j + 2] << "\t" << f1_M[9 * j + 3] << "\t" << f1_M[9 * j + 4]
					<< "\t" << f1_M[9 * j + 5] << "\t" << f1_M[9 * j + 6] << "\t" << f1_M[9 * j + 7] << "\t" << f1_M[9 * j + 8] << endl;

				fsA << x << "\t" << y << "\t" << f1_P[9 * j + 0] << "\t" << f1_P[9 * j + 1] << "\t" << f1_P[9 * j + 2] << "\t" << f1_P[9 * j + 3] << "\t" << f1_P[9 * j + 4]
					<< "\t" << f1_P[9 * j + 5] << "\t" << f1_P[9 * j + 6] << "\t" << f1_P[9 * j + 7] << "\t" << f1_P[9 * j + 8] << "\t" << endl;
			}
		}

		fsA << endl;

		fsA.close();
*/
		//////////////////////////////////////////////

		streaming << <gridsize, blocksize, 0, f_stream >> > (d_f1_P, d_f_P, XDIM, YDIM, ZDIM, it);												//PCL STREAMING STEP

		{																											// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}

		}

		streaming << <gridsize, blocksize, 0, f_stream >> > (d_f1_M, d_f_M, XDIM, YDIM, ZDIM, it);												//MUCUS STREAMING STEP

		{																											
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "mucus collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}

		}

		//////////////////////////////////////////////////////
/*
		cudaStatus = cudaMemcpy(f_M, d_f_M, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		cudaStatus = cudaMemcpy(f_P, d_f_P, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of u failed!\n");
		}

		outfile = raw_data + to_string(it) + "-fluid.dat";

		fsA.open(outfile.c_str(), ofstream::app);

		fsA << "streaming" << endl;


		for (j = 63 * 128 + 0; j < 64 * 128 + 128; j++)
		{
			if (j == 63 * 128 + 64 || j == 64 * 128 + 64)
			{
				int x = j%XDIM;
				int y = (j - j%XDIM) / XDIM;

				fsA << x << "\t" << y << "\t" << f_M[9 * j + 0] << "\t" << f_M[9 * j + 1] << "\t" << f_M[9 * j + 2] << "\t" << f_M[9 * j + 3] << "\t" << f_M[9 * j + 4]
					<< "\t" << f_M[9 * j + 5] << "\t" << f_M[9 * j + 6] << "\t" << f_M[9 * j + 7] << "\t" << f_M[9 * j + 8] << endl;

				fsA << x << "\t" << y << "\t" << f_P[9 * j + 0] << "\t" << f_P[9 * j + 1] << "\t" << f_P[9 * j + 2] << "\t" << f_P[9 * j + 3] << "\t" << f_P[9 * j + 4]
					<< "\t" << f_P[9 * j + 5] << "\t" << f_P[9 * j + 6] << "\t" << f_P[9 * j + 7] << "\t" << f_P[9 * j + 8] << "\t" << endl;
			}
		}

		fsA << endl;

		fsA.close();
		*/
		//////////////////////////////////////////////

		macro << <gridsize, blocksize, 0, f_stream >> > (d_f_P, d_f_M, d_rho_P, d_rho_M, d_rho, d_u, d_u_M, XDIM, YDIM, ZDIM);			//MACRO STEP

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "macro launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}

		}
		

		binaryforces << <gridsize, blocksize, 0, f_stream >> > (d_rho_P, d_rho_M, d_rho, d_f_P, d_f_M, d_force_P, d_force_M, d_force_E, d_u, d_u_M, XDIM, YDIM, ZDIM, G_PM, it);
		

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "binaryforces launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}
		

		cudaStreamWaitEvent(f_stream, cilia_done, 0);

		cudaEventSynchronize(cilia_done);
		
		cudaEventDestroy(cilia_done);

		

		interpolate << <gridsize2, blocksize2, 0, f_stream >> > (d_rho, d_u, Ns, d_u_s, d_F_s, d_s, XDIM, YDIM, ZDIM, d_nodes);						//IB INTERPOLATION STEP

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "interpolate launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		spread << <gridsize4, blocksize4, 0, f_stream >> > (Ns, d_u_s, d_F_s, d_force, d_s, XDIM, YDIM, ZDIM, d_epsilon, c_space, d_nodes);						//IB SPREADING STEP

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "spread launch failed: %s\n", cudaGetErrorString(cudaStatus));

				cout << it << endl;
				system("pause");
				return 1;
			}
		}

		forces << <gridsize, blocksize, 0, f_stream >> > (d_rho_P, d_rho_M, d_rho, d_f_P, d_f_M, d_force, d_force_P, d_force_M, d_u, d_Q, d_Q_P, d_Q_M, XDIM, YDIM, ZDIM);
		
		cudaEventRecord(fluid_done, f_stream);
		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "forces launch failed: %s\n", cudaGetErrorString(cudaStatus));
				
				cout << it << endl;
				system("pause");
				return 1;
			}

			cudaEventCreate(&Q_done);

			cudaStreamWaitEvent(o_stream, fluid_done, 0);

			cudaStatus = cudaMemcpyAsync(Q, d_Q, sizeof(double), cudaMemcpyDeviceToHost, o_stream);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of u failed!\n");
			}

			cudaStatus = cudaMemcpyAsync(Q_M, d_Q_M, sizeof(double), cudaMemcpyDeviceToHost, o_stream);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of u failed!\n");
			}

			cudaStatus = cudaMemcpyAsync(Q_P, d_Q_P, sizeof(double), cudaMemcpyDeviceToHost, o_stream);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of u failed!\n");
			}

			cudaEventRecord(Q_done, o_stream);
		}

		//----------------------------DATA OUTPUT------------------------------

		if (it % INTERVAL == 0)
		{
			{
				cudaEventSynchronize(fluid_done);

				cudaStatus = cudaMemcpy(rho, d_rho, size * sizeof(double), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "second cudaMemcpy of rho failed: %s\n", cudaGetErrorString(cudaStatus));
				}

				cudaStatus = cudaMemcpy(rho_P, d_rho_P, size * sizeof(double), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "second cudaMemcpy of rho_P failed: %s\n", cudaGetErrorString(cudaStatus));
				}

				cudaStatus = cudaMemcpy(rho_M, d_rho_M, size * sizeof(double), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "second cudaMemcpy of rho_M failed: %s\n", cudaGetErrorString(cudaStatus));
				}
			}

			if (BigData)
			{
				{

					cudaStatus = cudaMemcpy(u, d_u, 3 * size * sizeof(double), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy of u failed!\n");
					}

					cudaStatus = cudaMemcpy(force_P, d_force_P, 3 * size * sizeof(double), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy of force_P failed!\n");
					}

					cudaStatus = cudaMemcpy(force, d_force, 3 * size * sizeof(double), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy of force failed!\n");
					}

				}
				outfile = raw_data + to_string(it / INTERVAL) + "-fluid.dat";

				fsA.open(outfile.c_str());


				for (j = 0; j < size; j++)
				{
					int x = j % XDIM;
					int y = ((j - (j % XDIM)) / XDIM) % YDIM;
					int z = (j - j % (XDIM*YDIM)) / (XDIM*YDIM);

					float phi = (rho_P[j] - rho_M[j]) / (rho_P[j] + rho_M[j]);

					double ab = sqrt(u[0 * size + j] * u[0 * size + j] + u[1 * size + j] * u[1 * size + j]);

					double abforce = sqrt(force_M[0 * size + j] * force_M[0 * size + j] + force_M[1 * size + j] * force_M[1 * size + j]);

					fsA << x/**x_scale*/ << "\t" << y/**x_scale*/ << "\t" << z/**x_scale*/ << "\t" << rho[j] << "\t" << u[0 * size + j] * s_scale << "\t" << u[1 * size + j] * s_scale << "\t" << u[2 * size + j] * s_scale << "\t" << phi << endl;


					if (x == XDIM - 1) { fsA << endl; }
				}


				fsA.close();

				//cudaEventSynchronize(cilia_done);

				cudaStatus = cudaMemcpy(s, d_s, 3 * Ns * sizeof(float), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of s failed!\n"); }

				cudaStatus = cudaMemcpy(u_s, d_u_s, 3 * Ns * sizeof(float), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of u_s failed!\n"); }

				cudaStatus = cudaMemcpy(epsilon, d_epsilon, Ns * sizeof(float), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of epsilon failed!\n"); }

				cudaStatus = cudaMemcpy(F_s, d_F_s, 2 * Ns * sizeof(float), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of F_s failed!\n"); }

				outfile = cilia_data + to_string(it / INTERVAL) + "-cilia.dat";

				fsA.open(outfile.c_str());

				

					for (k = 0; k < Ns; k++)
					{
						fsA << s[3 * k + 0] << "\t" << s[3 * k + 1] << "\t" << s[3 * k + 2] << "\t" << u_s[3 * k + 0] * s_scale << "\t" << u_s[3 * k + 1] * s_scale << "\t" << F_s[2 * k + 0] << "\t" << F_s[2 * k + 1] << "\t" << epsilon[k] << "\n"; //LOOP FOR Np
						if (k % LENGTH == (LENGTH - 1) || s[3 * k + 0] > XDIM - 1 || s[3 * k + 0] < 1) fsA << "\n";
						//if (k == 1 && row == c_rows - 1) fsA << "\n";
					}
				
				fsA.close();

			}

			

			cudaEventSynchronize(Q_done);

			//fsB << it*t_scale << "\t" << Q[0] * x_scale << endl;

			//fsB.close();

			//------------------------density testing-------------------

			double density = 0;
			double PCL = 0.;
			double ML = 0.;
			double phimax = 0.;
			double phimin = 0.;

			for (int j = 0; j < size; j++)
			{
				float phi = (rho_P[j] - rho_M[j]) / (rho_P[j] + rho_M[j]);

				density += rho[j];
				PCL += rho_P[j];
				ML += rho_M[j];
				if (phi > phimax) phimax = phi;
				if (phi < phimin) phimin = phi;

			}
			density /= (size);
			PCL /= (size);
			ML /= (size);

			fsB.open(flux.c_str(), ofstream::app);

			fsB << it*t_scale << "\t" << density << "\t" << phimax << "\t" << phimin << "\t" << Q[0] * x_scale << "\t" << Q_P[0] * x_scale << "\t" << Q_M[0] * x_scale << endl;

			fsB.close();

			//----------------------------------------------------------
		}

		if (it == INTERVAL)
		{
			time_t cycle = seconds();

			p_runtime = (cycle - start)*(ITERATIONS / INTERVAL);

			time_t p_end = rawtime + p_runtime;

			timeinfo = localtime(&p_end);

			cout << "\nCompletion time: " << asctime(timeinfo) << endl;

			fsC << "\nCompletion time: " << asctime(timeinfo) << endl;

			if (p_runtime >= 345600. && ShARC)
			{
				int runtime_percent = int(345600. / p_runtime * 100.);

				fsC << "\nRuntime is longer than 4 days. ShARC will only run " << runtime_percent << "% of the simulation."<< endl;
			}


			fsC.close();
		}

	}

	cudaStreamDestroy(c_stream);
	cudaStreamDestroy(f_stream);
	cudaStreamDestroy(o_stream);
	
	double end = seconds();

	double runtime = end - start;

	int hours(0), mins(0);
	double secs(0.);

	if (runtime > 3600) hours = nearbyint(runtime / 3600 - 0.5);
	if (runtime > 60) mins = nearbyint((runtime - hours * 3600) / 60 - 0.5);
	secs = runtime - hours * 3600 - mins * 60;

	fsC.open(parameters.c_str(), ofstream::app);

	fsC << "Total runtime: ";
	if (hours < 10) fsC << "0";
	fsC << hours << ":";
	if (mins < 10) fsC << "0";
	fsC << mins << ":";
	if (secs < 10) fsC << "0";
	fsC << secs << endl;
	

	fsC.close();

	cudaDeviceReset();


	return 0;
}