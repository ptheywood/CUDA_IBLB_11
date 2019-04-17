#ifndef LATTICEBOLTZMANN_CUH
#define LATTICEBOLTZMANN_CUH

__global__ void equilibrium(const double * u, const double * rho, double * f0, const double * force, double * F, int XDIM, int YDIM, int ZDIM, double TAU);

__global__ void collision(const double * f0, const double * f, double * f1, const double * F, double TAU, int XDIM, int YDIM);

__global__ void streaming(const double * f1, double * f, int XDIM, int YDIM, int ZDIM, int it);

__global__ void macro(const double * f_P, const double * f_M, double * rho_P, double * rho_M, double * rho, double * u, double * u_M, int XDIM, int YDIM, int ZDIM);

__global__ void binaryforces(const double * rho_P, const double * rho_M, const double * rho, const double * f_P, const double * f_M, double * force_P, double * force_M, double * force_E, double * u, double * u_M, int XDIM, int YDIM, int ZDIM, const float G_PM, int it);

__global__ void forces(const double * rho_P, const double * rho_M, const double * rho, const double * f_P, const double * f_M, const double * force, double * force_P, double * force_M, double * u, double * Q, double * Q_P, double * Q_M, int XDIM, int YDIM, int ZDIM);
#endif // !LATTICEBOLTZMANN_H
