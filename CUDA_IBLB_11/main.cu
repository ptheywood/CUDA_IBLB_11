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

#define C_S 0.577		//SPEED OF SOUND ON LATTICE
#define RHO_0 1.		//FLUID DENSITY
#define PI 3.14159		//PI

//-------------------------------------------PARAMETER SCALING----------------------------

double l_0 = 0.000006;					//6 MICRON CILIUM LENGTH
double t_0 = 0.067;						//67ms BEAT PERIOD AT 15Hz


__global__ void define_filament(const int m, const int T, const int it, const double offset, double * s, double * lasts)
{
	int n(0);

	double arcl(0.);

	double a_n[2 * 7];
	double b_n[2 * 7];

	int threadnum = blockDim.x*blockIdx.x + threadIdx.x;

	int k = threadnum;

	double A_mn[7 * 2 * 3] =
	{
		-0.449,	 0.130, -0.169,	 0.063, -0.050, -0.040, -0.068,
		2.076, -0.003,	 0.054,	 0.007,	 0.026,	 0.022,	 0.010,
		-0.072, -1.502,	 0.260, -0.123,	 0.011, -0.009,	 0.196,
		-1.074, -0.230, -0.305, -0.180, -0.069,	 0.001, -0.080,
		0.658,	 0.793, -0.251,	 0.049,	 0.009,	 0.023, -0.111,
		0.381,	 0.331,	 0.193,	 0.082,	 0.029,	 0.002,	 0.048
	};

	double B_mn[7 * 2 * 3] =
	{
		0.0, -0.030, -0.093,  0.037,  0.062,  0.016, -0.065,
		0.0,  0.080, -0.044, -0.017,  0.052,  0.007,  0.051,
		0.0,  1.285, -0.036, -0.244, -0.093, -0.137,  0.095,
		0.0, -0.298,  0.513,  0.004, -0.222,  0.035, -0.128,
		0.0, -1.034,  0.050,  0.143,  0.043,  0.098, -0.054,
		0.0,  0.210, -0.367,  0.009,  0.120, -0.024,  0.102
	};

	{
		arcl = 1.*k / 10000;

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

		s[5 * (k + m * 10000) + 0] = 1. * 115 * a_n[2 * 0 + 0] * 0.5 + offset;
		s[5 * (k + m * 10000) + 1] = 1. * 115 * a_n[2 * 0 + 1] * 0.5;
		s[5 * (k + m * 10000) + 2] = 115 * arcl;

		for (n = 1; n < 7; n++)
		{
			s[5 * (k + m * 10000) + 0] += 1. * 115 * (a_n[2 * n + 0] * cos(n*2.*PI*it / T) + b_n[2 * n + 0] * sin(n*2.*PI*it / T));
			s[5 * (k + m * 10000) + 1] += 1. * 115 * (a_n[2 * n + 1] * cos(n*2.*PI*it / T) + b_n[2 * n + 1] * sin(n*2.*PI*it / T));
		}

		if (it > 0)
		{
			s[5 * (k + m * 10000) + 3] = s[5 * (k + m * 10000) + 0] - lasts[2 * (k + m * 10000) + 0];
			s[5 * (k + m * 10000) + 4] = s[5 * (k + m * 10000) + 1] - lasts[2 * (k + m * 10000) + 1];
		}
		

		lasts[2 * (k + m * 10000) + 0] = s[5 * (k + m * 10000) + 0];
		lasts[2 * (k + m * 10000) + 1] = s[5 * (k + m * 10000) + 1];
	}
}

__global__ void define_boundary(const int m, const int c_num, const double * boundary, double * b_points)
{
	int j(0), k(0);
	double b_length(0.);
	double step(1.);

	int threadnum = blockDim.x*blockIdx.x + threadIdx.x;

	k = threadnum;

	if (k == 0)
	{
		b_points[5 * (k + m * 100) + 0] = boundary[5 * (1 + m * 10000) + 0];
		b_points[5 * (k + m * 100) + 1] = boundary[5 * (1 + m * 10000) + 1];

		b_points[5 * (k + m * 100) + 2] = boundary[5 * (1 + m * 10000) + 3];
		b_points[5 * (k + m * 100) + 3] = boundary[5 * (1 + m * 10000) + 4];
	}
	else
	{
		b_length = k*step;

		for (j = (1 + m * 10000); j < c_num*10000; j++)
		{
			if (abs(boundary[5 * j + 2] - b_length) < 0.01)
			{
				b_points[5 * (k + m * 100) + 0] = boundary[5 * j + 0];
				b_points[5 * (k + m * 100) + 1] = boundary[5 * j + 1];

				b_points[5 * (k + m * 100) + 2] = boundary[5 * j + 3];
				b_points[5 * (k + m * 100) + 3] = boundary[5 * j + 4];

				j = c_num*10000;
			}
			else
			{
				b_points[5 * (k + m * 100) + 0] = 0.;
				b_points[5 * (k + m * 100) + 1] = 250.;

				b_points[5 * (k + m * 100) + 2] = 0.1;
				b_points[5 * (k + m * 100) + 3] = 0.1;
			}
		}
	}
}

void boundary_check(const int m, const double c_space, const int c_num, const int L, const double * s, int * epsilon)
{
	int r(0), k(0), l(0);

	int b_cross = 0;
	int lowest = 0;

	bool xclose = 0;
	bool yclose = 0;

	int r_max = 2 * L / c_space;

	double x_m(0.), y_m(0.), x_l(0.), y_l(0.);

	for (r = 1; r <= r_max; r++)
	{
		

		b_cross = 2 * L - r*c_space;

		if (b_cross > L) lowest = 0;
		else lowest = L - b_cross;

		for (k = lowest; k < L; k++)
		{
			x_m = s[2 * (k + m * 100) + 0];
			y_m = s[2 * (k + m * 100) + 1];

			for (l = lowest; l < L; l++)
			{
				xclose = 0;
				yclose = 0;

				if (m-r < 0)
				{
					x_l = s[2 * (l + (m - r + c_num) * 100) + 0];
					y_l = s[2 * (l + (m - r + c_num) * 100) + 1];
				}
				else
				{
					x_l = s[2 * (l + (m - r) * 100) + 0];
					y_l = s[2 * (l + (m - r) * 100) + 1];
				}

				if (abs(x_l - x_m) < 1) xclose = 1;

				if (abs(y_l - y_m) < 1) yclose = 1;

				if (xclose && yclose) epsilon[(k + m * 100)] = 0;

			}
		}
	}

}


int main(int argc, char * argv[])
{
	//----------------------------INITIALISING----------------------------

	unsigned int c_fraction = 1;
	unsigned int c_num = 6;
	double Re = 1.0;
	unsigned int XDIM = 300;
	unsigned int YDIM = 200;
	unsigned int T = 100000;
	unsigned int ITERATIONS = T;
	unsigned int INTERVAL = 100;
	unsigned int LENGTH = 100;
	bool ShARC = 0;
	bool BigData = 0;
	

	stringstream arg;

	arg << argv[1] << ' ' << argv[2] << ' ' << argv[3] << ' ' << argv[4] << ' ' << argv[5] << ' ' << argv[6] << ' ' << argv[7] << ' ' << argv[8];

	arg >> c_fraction >> c_num >> Re >> T >> ITERATIONS >> INTERVAL >> ShARC >> BigData;


	unsigned int c_space = LENGTH / 2;
	XDIM = c_num*c_space;
	const double centre[2] = { XDIM / 2., 0. };

	double dx = 1. / LENGTH;
	double dt = 1. / (T);
	double  SPEED = 0.8*1000/T;

	const double TAU = (SPEED*LENGTH) / (Re*C_S*C_S) + 1. / 2.;
	const double TAU2 = 1. / (12.*(TAU - (1. / 2.))) + (1. / 2.);

	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);

	cout << asctime(timeinfo) << endl;

	cout << "Initialising...\n";

	unsigned int i(0), j(0), k(0), n(0), m(0);

	unsigned int it(0);
	int phase(0);
	int p_step = T * c_fraction / c_num;

	
	double offset = 0.;

	double * lasts;
	lasts = new double[2 * c_num * 10000];

	double * boundary;
	boundary = new double[5 * c_num * 10000];

	int Np = 100 * c_num;
	double * b_points;

	b_points = new double[5 * Np];

	
	const int size = XDIM*YDIM;

	for (k = 0; k < c_num*10000; k++)
	{
		boundary[5 * k + 0] = 0.;
		boundary[5 * k + 1] = 0.;
		boundary[5 * k + 2] = 0.;
		boundary[5 * k + 3] = 0.;
		boundary[5 * k + 4] = 0.;

		lasts[2 * k + 0] = 0.;
		lasts[2 * k + 1] = 0.;

	}

	

	//-------------------------------CUDA PARAMETERS DEFINITION-----------------------


	int blocksize = 500;

	int gridsize = size / blocksize;

	int blocksize2 = c_num*LENGTH;

	int gridsize2 = 1;

	if (blocksize2 > 1000)
	{
		for (blocksize2 = 1000; blocksize2 > 0; blocksize2 -= LENGTH)
		{
			if ((c_num*LENGTH) % blocksize2 == 0)
			{
				gridsize2 = (c_num*LENGTH) / blocksize2;
				break;
			}
		}
	}

	cudaError_t cudaStatus;

	double Q = 0.;
	double E = 0.;

	if(ShARC) cudaStatus = cudaSetDevice(3);
	else cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Failed to set CUDA device.\n");
	}

	//------------------------------------------ERROR------------------------------------------------


	double l_error = (l_0*dx)*(l_0*dx);
	double t_error = (t_0*dt)*(t_0*dt);
	double c_error = (t_0*dt)*(t_0*dt) / ((l_0*dx)*(l_0*dx));
	double Ma = 1.*SPEED / C_S;
	time_t p_runtime;


	//-------------------------------------------ASSIGN CELL VALUES ON HEAP-----------------------------

	double * u;								//VELOCITY VECTOR

	u = new double[2 * size];

	double * rho;							//DENSITY

	rho = new double[size];

	double * f0;							//EQUILIBRIUM DISTRIBUTION FUNCTION

	f0 = new double[9 * size];

	double * f;								//DISTRIBUTION FUNCTION

	f = new double[9 * size];

	double * f1;							//POST COLLISION DISTRIBUTION FUNCTION

	f1 = new double[9 * size];

	double * force;							//MACROSCOPIC BODY FORCE VECTOR

	force = new double[2 * size];

	double * F;								//LATTICE BOLTZMANN FORCE

	F = new double[9 * size];

	unsigned int Ns = LENGTH * c_num;		//NUMBER OF BOUNDARY POINTS


	double * s;							//BOUNDARY POINTS

	double * u_s;						//BOUNDARY POINT VELOCITY

	double * F_s;						//BOUNDARY FORCE

	int * epsilon;

	s = new double[2 * Ns];

	u_s = new double[2 * Ns];

	F_s = new double[2 * Ns];

	epsilon = new int[Ns];

	for (k = 0; k < Ns; k++)
	{
		epsilon[k] = 1;
	}


	//----------------------------------------CREATE DEVICE VARIABLES-----------------------------

	double * d_u;								//VELOCITY VECTOR

	double * d_rho;							//DENSITY

	double * d_f0;							//EQUILIBRIUM DISTRIBUTION FUNCTION

	double * d_f;								//DISTRIBUTION FUNCTION

	double * d_f1;							//POST COLLISION DISTRIBUTION FUNCTION

	double * d_centre;

	double * d_force;

	double * d_F;

	double * d_F_s;

	double * d_s;

	double * d_u_s;

	int * d_epsilon;

	double * d_Q;

	

	double * d_lasts;

	double * d_boundary;

	double * d_b_points;



	//---------------------------CUDA MALLOC-------------------------------------------------------------
	{
		cudaStatus = cudaMalloc((void**)&d_u, 2 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_rho, size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_f0, 9 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_f, 9 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_f1, 9 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed,");
		}

		cudaStatus = cudaMalloc((void**)&d_centre, 2 * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_force, 2 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_F, 9 * size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&d_Q, sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

	}

	{

		cudaStatus = cudaMalloc((void**)&d_F_s, 2 * Ns * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of F_s failed!\n");
		}

		cudaStatus = cudaMalloc((void**)&d_s, 2 * Ns * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of s failed!\n");
		}

		cudaStatus = cudaMalloc((void**)&d_u_s, 2 * Ns * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u_s failed!\n");
		}

		cudaStatus = cudaMalloc((void**)&d_epsilon, Ns * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of epsilon failed!\n");
		}

		cudaStatus = cudaMalloc((void**)&d_lasts, 2 * c_num * 10000 * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u_s failed!\n");
		}

		cudaStatus = cudaMalloc((void**)&d_boundary, 5 * c_num * 10000 * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u_s failed!\n");
		}

		cudaStatus = cudaMalloc((void**)&d_b_points, 5 * Np * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc of u_s failed!\n");
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

	string cilia_data =  output_data + "Cilia/";
	cilia_data += to_string(c_num);
	cilia_data += "/";

	string img_data = output_data + "Img/";
	img_data += to_string(c_num);
	img_data += "/";
	
	img_data += to_string(c_num);

	string outfile = cilia_data;

	//----------------------------------------BOUNDARY INITIALISATION------------------------------------------------

	string flux = output_data + "/Flux/" + to_string(c_fraction) + "_" + to_string(c_num) +"-flux.dat";

	string parameters = raw_data + "/SimLog.txt";

	string input = "Data/cilium/";
	input += to_string(c_num);
	input += "/";

	ofstream fsA(input.c_str());

	ofstream fsB(flux.c_str());

	ofstream fsC(parameters.c_str());

	fsB.open(flux.c_str(), ofstream::trunc);

	fsB.close();

	fsC.open(parameters.c_str(), ofstream::trunc);

	fsC.close();


	//----------------------------------------INITIALISE ALL CELL VALUES---------------------------------------

	for (j = 0; j < XDIM*YDIM; j++)
	{
		rho[j] = RHO_0;
		u[2 * j + 0] = 0.0;
		u[2 * j + 1] = 0.0;

		force[2 * j + 0] = 0.;
		force[2 * j + 1] = 0.;


		for (i = 0; i < 9; i++)
		{
			f0[9 * j + i] = 0.;
			f[9 * j + i] = 0.;
			f1[9 * j + i] = 0.;
			F[9 * j + i] = 0.;
		}

	}

	//------------------------------------------------------COPY INITIAL VALUES TO DEVICE-----------------------------------------------------------

	//CUDA MEMORY COPIES
	{
		cudaStatus = cudaMemcpy(d_u, u, 2 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_rho, rho, size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_f0, f0, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_f, f, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_f1, f1, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_centre, centre, 2 * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_force, force, 2 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_F, F, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(d_lasts, lasts, 2 * c_num * 10000 * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of lasts failed!"); }

		cudaStatus = cudaMemcpy(d_boundary, boundary, 5 * c_num * 10000 * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of boundary failed!"); }


		cudaStatus = cudaMemcpy(d_Q, &Q, sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}


	}

	//------------------------------------------------------SET INITIAL DISTRIBUTION TO EQUILIBRIUM-------------------------------------------------

	equilibrium << <gridsize, blocksize >> > (d_u, d_rho, d_f0, d_force, d_F, XDIM, YDIM, TAU);				//INITIAL EQUILIBRIUM SET

	{																										// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "first equilibrium launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaMemcpy(f0, d_f0, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(F, d_F, 9 * size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}


	}

	for (j = 0; j < XDIM*YDIM; j++)
	{
		for (i = 0; i < 9; i++)
		{
			f[9 * j + i] = f0[9 * j + i];
		}
	}

	cudaStatus = cudaMemcpy(d_f, f, 9 * size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of f failed!\n");
	}



	//-----------------------------------------------------OUTPUT PARAMETERS------------------------------------------------------------------------


	fsC.open(parameters.c_str(), ofstream::trunc);

	fsC.close();

	fsC.open(parameters.c_str(), ofstream::app);

	fsC << asctime(timeinfo) << endl;
	fsC << "Size: " << XDIM << "x" << YDIM << endl;
	fsC << "Iterations: " << ITERATIONS << endl;
	fsC << "Reynolds Number: " << Re << endl;
	fsC << "Relaxation times: " << TAU << ", " << TAU2 << endl;
	//if (TAU <= 0.6) fsC << "POSSIBLE INSTABILITY! Relaxation time: " << TAU << endl;
	//if (TAU >= 2.01) fsC << "POSSIBLE INACCURACY! Relaxation time: " << TAU << endl;

	fsC << "Spatial step: " << dx*l_0 << "m" << endl;
	fsC << "Time step: " << dt*t_0 << "s" << endl;
	fsC << "Mach number: " << Ma << endl;
	fsC << "Spatial discretisation error: " << l_error << endl;
	fsC << "Time discretisation error: " << t_error << endl;
	fsC << "Compressibility error: " << c_error << endl;


	fsC << "\nThreads per block: " << blocksize << endl;
	fsC << "Blocks: " << gridsize << endl;


	//--------------------------ITERATION LOOP-----------------------------
	cout << "Running Simulation...\n";

	time_t start = seconds();

	for (it = 0; it < ITERATIONS; it++)
	{
	
		//--------------------------CILIA BEAT DEFINITION-------------------------

		
			for (m = 0; m < c_num; m++)
			{
				if (it + m*p_step == T) phase = T;
				else phase = (it + m*p_step) % T;

				offset = 1.*(m - (c_num - 1) / 2.)*c_space;


				define_filament << <10, 1000 >> > (m, T, phase, offset, d_boundary, d_lasts);

				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "define_filament failed: %s\n", cudaGetErrorString(cudaStatus)); }

				define_boundary << <1, 100 >> > (m, c_num, d_boundary, d_b_points);

				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "define_boundary failed: %s\n", cudaGetErrorString(cudaStatus)); }


				cudaStatus = cudaMemcpy(b_points, d_b_points, 5 * Np * sizeof(double), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy of b_points failed!\n"); }

			}

		
			for (j = 0; j < c_num*LENGTH; j++)
			{
				k = j;

				s[2 * k + 0] = n*LENGTH / 2.*c_num + (LENGTH / 2.*c_num) / 2. + b_points[5 * j + 0];

				if (s[2 * k + 0] < 0) s[2 * k + 0] += XDIM;
				else if (s[2 * k + 0] > XDIM) s[2 * k + 0] -= XDIM;

				s[2 * k + 1] = b_points[5 * j + 1];

				if (it == 0)
				{
					u_s[2 * k + 0] = 0.;
					u_s[2 * k + 1] = 0.;
				}
				else
				{
					u_s[2 * k + 0] = b_points[5 * j + 2];
					u_s[2 * k + 1] = b_points[5 * j + 3];
				}

				E += u_s[2 * k + 0];

				epsilon[k] = 1;
			}
		
		
		for (m = 0; m < c_num; m++)
		{
			boundary_check(n*c_num + m, c_space, c_num, LENGTH, s, epsilon);
		}
		
		//---------------------------CILIUM COPY---------------------------------------- 

		{

			cudaStatus = cudaMemcpy(d_epsilon, epsilon, Ns * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of epsilon failed!\n");
			}

			cudaStatus = cudaMemcpy(d_s, s, 2 * Ns * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of s failed!\n");
			}

			cudaStatus = cudaMemcpy(d_u_s, u_s, 2 * Ns * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of u_s failed!\n");
			}

			cudaStatus = cudaMemcpy(d_F_s, F_s, 2 * Ns * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of F_s failed!\n");
			}
		}


		//---------------------------IMMERSED BOUNDARY LATTICE BOLTZMANN STEPS-------------------

		equilibrium << <gridsize, blocksize >> > (d_u, d_rho, d_f0, d_force, d_F, XDIM, YDIM, TAU);				//EQUILIBRIUM STEP

		{																										// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "equilibrium launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		collision << <gridsize, blocksize >> > (d_f0, d_f, d_f1, d_F, TAU, TAU2, XDIM, YDIM, it);						//COLLISION STEP

		{																										// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		streaming << <gridsize, blocksize >> > (d_f1, d_f, XDIM, YDIM);												//STREAMING STEP

		{																											// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}

		}

		macro << <gridsize, blocksize >> > (d_f, d_u, d_rho, XDIM, YDIM);											//MACRO STEP

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "collision launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		interpolate << <gridsize2, blocksize2 >> > (d_rho, d_u, Ns, d_u_s, d_F_s, d_s, XDIM);											//IB INTERPOLATION STEP

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "interpolate launch failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}

		spread << <gridsize, blocksize >> > (d_rho, d_u, d_f, Ns, d_u_s, d_F_s, d_force, d_s, XDIM, d_Q, d_epsilon);	//IB SPREADING STEP

		{
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "spread launch failed: %s\n", cudaGetErrorString(cudaStatus));
				
				//cout << it << endl;
				//system("pause");
				return 1;
			}

			cudaStatus = cudaMemcpy(rho, d_rho, size * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of rho failed!\n");
			}

			cudaStatus = cudaMemcpy(u, d_u, 2 * size * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of u failed!\n");
			}

			cudaStatus = cudaMemcpy(&Q, d_Q, sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy of u failed!\n");
			}
		}

		//----------------------------DATA OUTPUT------------------------------


		double t_scale = 1000.*dt*t_0;					//milliseconds
		double x_scale = 1000000. * dx*l_0;				//microns
		double s_scale = 1000.*x_scale / t_scale;		//millimetres per second


		if (it % INTERVAL == 0)
		{
			if (BigData)
			{
				outfile = raw_data + to_string(it) + "-fluid.dat";

				fsA.open(outfile.c_str());

				for (j = 0; j < XDIM*YDIM; j++)
				{
					int x = j%XDIM;
					int y = (j - j%XDIM) / XDIM;

					double ab = sqrt(u[2 * j + 0] * u[2 * j + 0] + u[2 * j + 1] * u[2 * j + 1]);

					fsA << x*x_scale << "\t" << y*x_scale << "\t" << u[2 * j + 0]*s_scale << "\t" << u[2 * j + 1]*s_scale << "\t" << ab*s_scale << "\t" << rho[j] << endl;


					if (x == XDIM - 1) fsA << endl;
				}

				fsA.close();

				outfile = cilia_data + to_string(it) + "-cilia.dat";

				fsA.open(outfile.c_str());

				for (k = 0; k < Ns; k++)
				{
					fsA << s[2 * k + 0] << "\t" << s[2 * k + 1] << "\t" << u_s[2 * k + 0] << "\t" << u_s[2 * k + 1] << "\t" << epsilon[k] << "\n"; //LOOP FOR Np
					if (k % 100 == 99 || s[2 * k + 0] > XDIM - 1 || s[2 * k + 0] < 1) fsA << "\n";
				}

				fsA.close();

			}
			
			fsB.open(flux.c_str(), ofstream::app);

			fsB << it*1000.*dt*t_0 << "\t" << Q*1000000. * dx*l_0*1000000. * dx*l_0 << "\t" << E*1000000. * dx*l_0*1000000. * dx*l_0 << endl;

			fsB.close();
		}

		if (it == INTERVAL)
		{
			time_t cycle = seconds();

			p_runtime = (cycle - start)*(ITERATIONS / INTERVAL);

			time_t p_end = rawtime + p_runtime;

			timeinfo = localtime(&p_end);

			//int hours(0), mins(0);
			//time_t secs(0.);

			//if (p_runtime >= 3600) hours = nearbyint(p_runtime / 3600 - 0.5);
			//if (p_runtime >= 60) mins = nearbyint((p_runtime - hours * 3600) / 60 - 0.5);
			//secs = p_runtime - hours * 3600 - mins * 60;

			//cout << "\nProjected runtime: ";
			//if (hours < 10) cout << 0;
			//cout << hours << ":";
			//if (mins < 10) cout << 0;
			//cout << mins << ":";
			//if (secs < 10) cout << 0;
			//cout << fixed << setprecision(2) << secs;

			cout << "\nCompletion time: " << asctime(timeinfo) << endl;

			fsC << "\nCompletion time: " << asctime(timeinfo) << endl;

			fsC.close();
		}

	}

	fsB.open(flux.c_str(), ofstream::app);

	fsB << it*1000.*dt*t_0 << "\t" << Q*1000000. * dx*l_0*1000000. * dx*l_0 << endl;

	fsB.close();
	
	double end = seconds();

	double runtime = end - start;

	int hours(0), mins(0);
	double secs(0.);

	if (runtime > 3600) hours = nearbyint(runtime / 3600 - 0.5);
	if (runtime > 60) mins = nearbyint((runtime - hours * 3600) / 60 - 0.5);
	secs = runtime - hours * 3600 - mins * 60;

	fsC.open(parameters.c_str(), ofstream::app);

	fsC << "\nTotal runtime: ";
	if (hours < 10) fsC << 0;
	fsC << hours << ":";
	if (mins < 10) fsC << 0;
	fsC << mins << ":";
	if (secs < 10) fsC << 0;
	fsC << secs << endl;
	fsC << "Net Q = " << Q << " Avg Q = " << Q / 1.*(it / T) << endl;

	fsC.close();

	cudaDeviceReset();


	return 0;
}