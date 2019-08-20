#include "raytrace.hh"

#include <cassert>
#include <iostream>
#include "helper_cuda.h"
#include "helper_math.h"

__device__ float siddonRPL(
        float3 dest,
        float3 source,
        float3 start,
        uint3 size,
        float3 spacing,
        cudaTextureObject_t texDens,
        float stop_early=-1.f
) {
    // avoid division by zero (fixes blank center pixel issue)
    dest += 1e-9f;

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
        int max_iters = 5000;
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

            if (intersection>=0.001f) { // do not process unless > 0.1 mm
                alpha_mid = (nextalpha + alpha)*0.5f; // midpoint between intersections
                // Remember that this function traces only a single ray.
                // rpl has been set to zero during initialisation.
                float fetchX = (source.x + alpha_mid*diff.x - start.x) / spacing.x;
                float fetchY = (source.y + alpha_mid*diff.y - start.y) / spacing.y;
                float fetchZ = (source.z + alpha_mid*diff.z - start.z) / spacing.z;

                rpl += intersection * tex3D<float>( texDens, fetchX+0.5f, fetchY+0.5f, fetchZ+0.5f);
                if (stop_early >= 0 && rpl > stop_early) { break; }
            }
            alpha = nextalpha;
        }
        assert(iter<max_iters); // something is wrong
    }
    if (!isfinite(rpl)) { rpl = 0.f; }
    return rpl;
}

#define subvoxidx   (threadIdx.x)
#define subvoxcount (blockDim.x)
#define s_index(ii) threadIdx.y+blockDim.y*ii
// Ray-trace from source along beamlet central axis, returning path-length
__global__ void cudaRayTrace(
        float*  rpl,         // output of radiologic path lengths for each src/dest pair
        float*  dests,       // array of dest coords
        float3  source,      // common src coords
        uint    npts,        // number of dests/rpls
        float3  densStart,   // coords of volume start
        uint3   densSize,    // array size
        float3  densSpacing, // voxelsize of volume
        cudaTextureObject_t texDens,    // volume texture with tri-linear interpolation
        float   stop_early=-1.f
) {
    // supersampling setup
    extern __shared__ float s_subterma[];
    s_subterma[s_index(subvoxidx)] = 0.f;
    __syncthreads();

    uint tid = threadIdx.y + blockIdx.x * blockDim.y;

    if (tid < npts) {
        // for each voxel, the ray begins at the beam source (point source assumption)
        // the ray ends at the center of current voxel
        float3 dest = {dests[3*tid], dests[3*tid + 1], dests[3*tid + 2]};
        if (subvoxcount > 1) {
            // get sub-vox center position
            int ss_factor = cbrtf(subvoxcount);
            int sub_X = subvoxidx % ss_factor;
            int sub_Y = (subvoxidx / ss_factor)%ss_factor;
            int sub_Z = subvoxidx / (ss_factor*ss_factor);

            float3 subvoxsize = densSpacing/(float)ss_factor;
            dest = make_float3(
                    dest.x + (sub_X+0.5f)*subvoxsize.x,
                    dest.y + (sub_Y+0.5f)*subvoxsize.y,
                    dest.z + (sub_Z+0.5f)*subvoxsize.z );
        }

        // radiological path length
        s_subterma[s_index(subvoxidx)] = siddonRPL(
                    dest,
                    source,
                    densStart,
                    densSize,
                    densSpacing,
                    texDens,
                    stop_early
                    );
    }

    // -------------------------
    __syncthreads();
    float avgterma = 0.f;
    if (tid<npts && subvoxidx==0) {
        for (int ii=0; ii<subvoxcount; ii++) {
            avgterma += s_subterma[s_index(ii)];
        }
        rpl[tid] = (avgterma/subvoxcount);
    }
}


// C-CALLABLE ENTRY POINT
void raytrace_c(
    float* rpl,
    float* dests,
    float3 source,
    uint   npts,
    float* dens,
    float3 densStart,
    uint3  densSize,
    float3 densSpacing,
    std::ostream& cout,
    float  stop_early,
    uint   ssfactor
) {
    /* // Force supersampling off */
    /* ssfactor = 1; */
    //
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
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t texDens;
    checkCudaErrors( cudaCreateTextureObject(&texDens, &resDesc, &texDesc, NULL) );

    // allocate device memory
    float* d_rpl;
    checkCudaErrors( cudaMalloc(&d_rpl, npts*sizeof(float)) );
    float* d_dests;
    checkCudaErrors( cudaMalloc(&d_dests, npts*3*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(d_dests, dests, npts*3*sizeof(float), cudaMemcpyHostToDevice) );

    // setup kernel call params
    dim3 rayBlock;
    uint threadLimit = 256;
    if (ssfactor>1) {
        uint nsubvoxels = powf(ssfactor, 3);
        rayBlock = dim3(nsubvoxels, (uint)floorf((float)threadLimit/nsubvoxels), 1);
    } else {
        rayBlock = dim3(1, threadLimit, 1);
    }
    dim3 rayGrid = dim3((uint)ceilf((float)npts/rayBlock.y), 1, 1);
    int sharedMem = rayBlock.x*rayBlock.y*rayBlock.z*sizeof(float);
    /* cout << "rayBlock: ("<<rayBlock.x<<","<<rayBlock.y<<","<<rayBlock.z<<")"<<std::endl; */
    /* cout << "rayGrid: ("<<rayGrid.x<<","<<rayGrid.y<<","<<rayGrid.z<<")"<<std::endl; */
    /* cout << "raySharedMem: "<<sharedMem << " bytes"<< std::endl; */

    // call raytracing kernel
    cudaRayTrace <<< rayGrid, rayBlock, sharedMem >>>(
            d_rpl,
            d_dests,
            source,
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

    // cout << "npts: " << npts << std::endl;
    // cout << "stop_early: " << stop_early << std::endl;
    // cout << "src: " << source.x << " " << source.y << " " << source.z << std::endl;
    // cout << "start: " << densStart.x << " " << densStart.y << " " << densStart.z << std::endl;
    // cout << "size: " << densSize.x << " " << densSize.y << " " << densSize.z << std::endl;
    // cout << "spacing: " << densSpacing.x << " " << densSpacing.y << " " << densSpacing.z << std::endl;
}
