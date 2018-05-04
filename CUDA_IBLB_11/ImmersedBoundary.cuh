#ifndef IMMERSEDBOUNDARY_CUH
#define IMMERSEDBOUNDARY_CUH

__device__ float d_delta(const float & xs, const float & ys, const int & x, const int & y);

__global__ void interpolate(const double * rho, const double * u, const int Ns, const float * u_s, float * F_s, const float * s, const int XDIM, const int YDIM);

__global__ void spread(const double * rho, double * u, const double * f, const int Ns, const float * u_s, const float * F_s, double * force, const float * s, const int XDIM, double * Q, const int * epsilon);


#endif // !IMMERSEDBOUNDARY_H