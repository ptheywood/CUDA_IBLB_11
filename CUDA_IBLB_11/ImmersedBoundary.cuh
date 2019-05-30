#ifndef IMMERSEDBOUNDARY_CUH
#define IMMERSEDBOUNDARY_CUH

__device__ float d_delta(const float & xs, const float & ys, const float & zs, const int & x, const int & y, const int & z);

__global__ void interpolate(const double * rho, const double * u, const int Ns, const float * u_s, float * F_s, const float * s, const int XDIM, const int YDIM, const int ZDIM, int * nodes);

__global__ void spread(const int Ns, const float * u_s, const float * F_s, double * force, const float * s, const int XDIM, const int YDIM, const int ZDIM, const int * epsilon, const int c_space, int * nodes);


#endif // !IMMERSEDBOUNDARY_H