#include "raytrace.h"

#include <cassert>
#include <iostream>
#include "helper_cuda.h"
#include "helper_math.h"
#include "geometry.cuh"

__device__ float siddonRPL(
    float3 source,
    float3 dest,
    float3 start,
    uint3 size,
    float3 spacing,
    cudaTextureObject_t texDens,
    float stop_early=-1.f
) {
  // avoid division by zero (fixes blank center pixel issue)
  dest += 1e-12f;

  // projection vector for this thread
  float3 diff = dest - source;
  ////////////////////////////////////////////////////////

  /* Siddon's algorithm normalizes the distance between points 1 and 2
   * alpha is the current position along that normalized vector as a scalar amount
   * alpha_min and alpha_max set the bounds of intersection between this vector and the volume of interest
   * limits of alpha parameter - aligned to plane positions not voxel center
   */

  float a_min, a_max;
  float3 alpha_min, alpha_max;
  {
    float3 end = start + spacing*make_float3(size);

    float3 a_first, a_last;
    a_first.x = (start.x - source.x - 0.5f*spacing.x) / diff.x;
    a_first.y = (start.y - source.y - 0.5f*spacing.y) / diff.y;
    a_first.z = (start.z - source.z - 0.5f*spacing.z) / diff.z;
    a_last.x  = (end.x   - source.x + 0.5f*spacing.x) / diff.x;
    a_last.y  = (end.y   - source.y + 0.5f*spacing.y) / diff.y;
    a_last.z  = (end.z   - source.z + 0.5f*spacing.z) / diff.z;

    alpha_min.x = min(a_first.x, a_last.x);
    alpha_min.y = min(a_first.y, a_last.y);
    alpha_min.z = min(a_first.z, a_last.z);
    alpha_max.x = max(a_first.x, a_last.x);
    alpha_max.y = max(a_first.y, a_last.y);
    alpha_max.z = max(a_first.z, a_last.z);

    a_min = fmaxf(0,fmaxf(fmaxf(alpha_min.x, alpha_min.y), alpha_min.z));
    a_max = fminf(1,fminf(fminf(alpha_max.x, alpha_max.y), alpha_max.z));
  }

  float rpl = 0.f;
  if (!(a_min >= a_max)) {
    float d12 = length(diff);   // distance between ray end points
    float step_x = fabsf(spacing.x/diff.x);
    float step_y = fabsf(spacing.y/diff.y);
    float step_z = fabsf(spacing.z/diff.z);

    // step along the vector, sampling each time we cross any plane (meaning we've entered a new voxel)
    float alpha   = a_min;
    float alpha_x = a_min;
    float alpha_y = a_min;
    float alpha_z = a_min;
    float nextalpha;
    float alpha_mid;
    const int max_iters = 5000;
    const float intersect_min = 0.001f*fmin(fmin(spacing.x, spacing.y), spacing.z);
    int iter = 0;
    while (alpha < a_max && ++iter < max_iters) {
      // find next intersection plane
      bool valid_x, valid_y, valid_z;
      if (alpha_x >= 0.0f && alpha_x < 1.0f) { valid_x = true; } else { valid_x = false; };
      if (alpha_y >= 0.0f && alpha_y < 1.0f) { valid_y = true; } else { valid_y = false; };
      if (alpha_z >= 0.0f && alpha_z < 1.0f) { valid_z = true; } else { valid_z = false; };
      if (!(valid_x || valid_y || valid_z)) { break; }
      if (valid_x && (!valid_y || alpha_x <= alpha_y) && (!valid_z || alpha_x <= alpha_z)) {
        nextalpha = alpha_x;
        alpha_x += step_x;
      }
      else if (valid_y && (!valid_x || alpha_y <= alpha_x) && (!valid_z || alpha_y <= alpha_z)) {
        nextalpha = alpha_y;
        alpha_y += step_y;
      }
      else if (valid_z && (!valid_x || alpha_z <= alpha_x) && (!valid_y || alpha_z <= alpha_y)) {
        nextalpha = alpha_z;
        alpha_z += step_z;
      }

      // the total intersection length of previous voxel
      float intersection = fabsf(d12*(nextalpha-alpha)); // intersection is voxel intersection length

      if (intersection>=intersect_min) { // do not process unless > 0.1 mm
        alpha_mid = (nextalpha + alpha)*0.5f; // midpoint between intersections
        // Remember that this function traces only a single ray.
        // rpl has been set to zero during initialisation.
        float fetchX = (source.x + alpha_mid*diff.x - start.x) / spacing.x;
        float fetchY = (source.y + alpha_mid*diff.y - start.y) / spacing.y;
        float fetchZ = (source.z + alpha_mid*diff.z - start.z) / spacing.z;

        rpl += intersection * tex3D<float>( texDens, fetchX, fetchY, fetchZ);
        if (stop_early >= 0 && rpl > stop_early) { break; }
      }
      alpha = nextalpha;
    }
    assert(iter<max_iters); // something is wrong
  }
  if (!isfinite(rpl)) { rpl = 0.f; }
  return rpl;
}

// Ray-trace from source along beamlet central axis, returning path-length
__global__ void cudaRayTrace(
    float*  rpl,                  // output of radiologic path lengths for each src/dest pair
    float*  sources,              // array of src coords
    float*  dests,                // array of dest coords
    uint    npts,                 // number of dests/rpls
    float3  densStart,            // coords of volume start
    uint3   densSize,             // array size
    float3  densSpacing,          // voxelsize of volume
    cudaTextureObject_t texDens,  // volume texture with tri-linear interpolation
    float   stop_early=-1.f
) {
  uint tid = threadIdx.y + blockIdx.x * blockDim.y;

  if (tid < npts) {
    // for each voxel, the ray begins at the beam source (point source assumption)
    // the ray ends at the center of current voxel
    float3 source = {sources[3*tid], sources[3*tid + 1], sources[3*tid + 2]};
    float3 dest = {dests[3*tid], dests[3*tid + 1], dests[3*tid + 2]};

    // radiological path length
    rpl[tid] = siddonRPL(
        source,
        dest,
        densStart,
        densSize,
        densSpacing,
        texDens,
        stop_early
        );
  }
}

// Ray-trace from common source through center/corners/mid-edges of "detector" element positioned on the detector plane which is centered at isocenter and oriented at some angle
#define subrayidx   (threadIdx.x)
#define subraycount (blockDim.x)
#define s_index(ii) threadIdx.y+blockDim.y*(threadIdx.z+blockDim.z*ii)
__global__ void cudaBeamTrace(
        float* rpl,
        float  sad,
        uint2  detDims,
        float3 detCenter,
        float2 detSpacing,
        float2 detPixelSize,
        float  detAzi, // gantry angle
        float  detZen, // couch angle
        float  detAng, // detAngimator angle
        float3 densStart,
        uint3  densSize,
        float3 densSpacing,
        cudaTextureObject_t texDens,
        float stop_early=-1.f
) {
	// dynamically allocated shared memory
    extern __shared__ char s_subrpl[];
    s_subrpl[s_index(subrayidx)] = 0;
    __syncthreads();

    int ray_X = threadIdx.y + blockIdx.y*blockDim.y;
    int ray_Z = threadIdx.z + blockIdx.z*blockDim.z;

    // threads are launched for each pixel in the 2D output map
    // ray_X/ray_Z in Fluence Coord Sys (FCS) which matches RCS before rotation
    if (ray_X >= detDims.x || ray_Z >= detDims.y) { return; }

    // shift for sub-ray [-1, 0, +1]
    int subray_X = (subrayidx % 3) - 1;
    int subray_Z = (subrayidx / 3) - 1;

	  // the end point of each ray is found from the 2D coordinates on the fluence map
    // center coord of fluence map is detCenter
    // bixel coords defined in ray_X-ray_Z plane (FCS) then rotated with beam angles into RCS
    // dont let x/y in int2 scare you, it is really x/z
    // we directly define bixel coords in DCS to be consistent with storage of texRay
    float2 detsize = make_float2(detDims-1)*detSpacing;
    float3 bixel_ctr_FCS = make_float3(
            -0.5f*detsize.x + ray_X*detSpacing.x + 0.5f*subray_X*detPixelSize.x,
            0,
            -0.5f*detsize.y + ray_Z*detSpacing.y + 0.5f*subray_Z*detPixelSize.y
            );

    float3 bixel_ctr = inverseRotateBeamAtOriginRHS(bixel_ctr_FCS, detAzi, detZen, detAng);
    bixel_ctr += detCenter;

    float3 source = inverseRotateBeamAtOriginRHS(make_float3(0.f, -sad, 0.f), detAzi, detZen, detAng) + detCenter;

    // extend end of raytrace beyond fluence map plane
    float3 shortdiff = bixel_ctr - source;
    float3 sink = source + 10.f*shortdiff;

	// the vector of projection for this thread, extended completely through volume
    float3 diff = sink - source;

    s_subrpl[s_index(subrayidx)] = siddonRPL(
            source,
            sink,
            densStart,
            densSize,
            densSpacing,
            texDens,
            stop_early
            );

    __syncthreads();
    if (subrayidx == 0)  {
        float f = 0.f;
        for (int ii=0; ii<subraycount; ii++) {
            f += s_subrpl[s_index(ii)];
        }

        // write out the fluence map
        rpl[ray_X + detDims.x * ray_Z] = f/9.f;
    }
}

// C-CALLABLE ENTRY POINT
void raytrace_c(
    float* rpl,
    const float* sources,
    const float* dests,
    unsigned int   npts,
    float* dens,
    float3 densStart,
    uint3  densSize,
    float3 densSpacing,
    float  stop_early
) {
  // allocate cudaArray opaque memory
  cudaArray* d_densArr;
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaExtent extent = make_cudaExtent(densSize.x, densSize.y, densSize.z);
  checkCudaErrors( cudaMalloc3DArray(&d_densArr, &desc, extent) );

  // copy to cudaArray
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr = make_cudaPitchedPtr(dens, extent.width*sizeof(float), extent.width, extent.height);
  copyParams.dstArray = d_densArr;
  copyParams.kind = cudaMemcpyHostToDevice;
  copyParams.extent = extent;
  checkCudaErrors( cudaMemcpy3D(&copyParams) );

  // create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(cudaResourceDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = d_densArr;
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(cudaTextureDesc));
  texDesc.normalizedCoords = false;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.readMode = cudaReadModeElementType;
  cudaTextureObject_t texDens;
  checkCudaErrors( cudaCreateTextureObject(&texDens, &resDesc, &texDesc, NULL) );

  // allocate device memory
  float* d_rpl;
  checkCudaErrors( cudaMalloc(&d_rpl, npts*sizeof(float)) );

  float* d_sources;
  checkCudaErrors( cudaMalloc(&d_sources, npts*sizeof(float3)) );
  checkCudaErrors( cudaMemcpy(d_sources, sources, npts*3*sizeof(float), cudaMemcpyHostToDevice) );

  float* d_dests;
  checkCudaErrors( cudaMalloc(&d_dests, npts*3*sizeof(float)) );
  checkCudaErrors( cudaMemcpy(d_dests, dests, npts*3*sizeof(float), cudaMemcpyHostToDevice) );

  // setup kernel call params
  dim3 rayBlock;
  uint threadLimit = 256;
  rayBlock = dim3(1, threadLimit, 1);
  dim3 rayGrid = dim3((uint)ceilf((float)npts/rayBlock.y), 1, 1);
  int sharedMem = rayBlock.x*rayBlock.y*rayBlock.z*sizeof(float);
  // std::cout << "rayBlock: ("<<rayBlock.x<<","<<rayBlock.y<<","<<rayBlock.z<<")"<<std::endl;
  // std::cout << "rayGrid: ("<<rayGrid.x<<","<<rayGrid.y<<","<<rayGrid.z<<")"<<std::endl;
  // std::cout << "raySharedMem: "<<sharedMem << " bytes"<< std::endl;

  // call raytracing kernel
  cudaRayTrace <<< rayGrid, rayBlock, sharedMem >>>(
      d_rpl,
      d_sources,
      d_dests,
      npts,
      densStart,
      densSize,
      densSpacing,
      texDens,
      stop_early
      );
  cudaDeviceSynchronize();
  getLastCudaError("Error during cudaRayTrace");
  checkCudaErrors( cudaMemcpy(rpl, d_rpl, npts*sizeof(float), cudaMemcpyDeviceToHost) );

  // free memory
  checkCudaErrors( cudaFree(d_rpl) );
  checkCudaErrors( cudaFree(d_dests) );
  checkCudaErrors( cudaDestroyTextureObject(texDens) );
  checkCudaErrors( cudaFreeArray(d_densArr) );
}

void beamtrace_c(
  float*        rpl,
  const float   sad,
  const uint2   detDims,
  const float3  detCenter,
  const float2  detSpacing,
  const float2  detPixelSize,
  const float   detAzi,
  const float   detZen,
  const float   detAng,
  const float*  dens,
  const float3  densStart,
  const uint3   densSize,
  const float3  densSpacing,
  const float   stop_early
) {
  // allocate cudaArray opaque memory
  cudaArray* d_densArr;
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaExtent extent = make_cudaExtent(densSize.x, densSize.y, densSize.z);
  checkCudaErrors( cudaMalloc3DArray(&d_densArr, &desc, extent) );

  // copy to cudaArray
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr = make_cudaPitchedPtr((void*)dens, extent.width*sizeof(float), extent.width, extent.height);
  copyParams.dstArray = d_densArr;
  copyParams.kind = cudaMemcpyHostToDevice;
  copyParams.extent = extent;
  checkCudaErrors( cudaMemcpy3D(&copyParams) );

  // create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(cudaResourceDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = d_densArr;
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(cudaTextureDesc));
  texDesc.normalizedCoords = false;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.readMode = cudaReadModeElementType;
  cudaTextureObject_t texDens;
  checkCudaErrors( cudaCreateTextureObject(&texDens, &resDesc, &texDesc, NULL) );
  //////////////////////////////////////////////////////////////////////////////////////////////

  // projection through iso_cntr_matrix to create conformal field map from source
  float *d_rpl;
  int nbixels = detDims.x * detDims.y;
  checkCudaErrors( cudaMalloc( &d_rpl, nbixels*sizeof(float) ) );
  checkCudaErrors( cudaMemset( d_rpl, 0, nbixels*sizeof(float) ) );

  // use center/corner/edge sampling pattern
  dim3 rayGrid;
  dim3 rayBlock = dim3{9, 8, 8};
  // create a thread for each pixel in a 2D square fluence map array
  rayGrid.y = ceilf((float)detDims.x/rayBlock.y);
  rayGrid.z = ceilf((float)detDims.y/rayBlock.z);

  // call raytracing kernel
  size_t sharedMem = rayBlock.x*rayBlock.y*rayBlock.z*sizeof(char);
  cudaBeamTrace <<< rayGrid, rayBlock, sharedMem >>>(
      d_rpl,
      sad,
      detDims,
      detCenter,
      detSpacing,
      detPixelSize,
      detAzi,
      detZen,
      detAng,
      densStart,
      densSize,
      densSpacing,
      texDens,
      stop_early
      );
  cudaDeviceSynchronize();
  getLastCudaError("Error during cudaBeamTrace");
  checkCudaErrors( cudaMemcpy(rpl, d_rpl, nbixels*sizeof(float), cudaMemcpyDeviceToHost) );

  // free memory
  checkCudaErrors( cudaDestroyTextureObject(texDens) );
  checkCudaErrors( cudaFreeArray(d_densArr) );
  checkCudaErrors( cudaFree(d_rpl) );
}
