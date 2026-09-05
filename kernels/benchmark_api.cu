// SPDX-License-Identifier: Apache-2.0
// Minimal research harness ABI: device pointers are owned by the Python caller.
// This is not a production operator or a sound-summary construction interface.
#include "moments.cu"

namespace cmk {
__global__ void query_shifts(SummaryView s, const double* q, int Q, double* shifts) {
  int qi = blockIdx.x;
  if (qi >= Q) return;
  __shared__ double maxima[128];
  double best = -INFINITY;
  for (int b = threadIdx.x; b < s.B; b += blockDim.x) {
    double score = 0;
    for (int a = 0; a < s.d; ++a) score += q[qi * s.d + a] * s.mu[b * s.d + a];
    best = fmax(best, score + log(s.count[b]));
  }
  maxima[threadIdx.x] = best;
  __syncthreads();
  for (int width = 64; width; width /= 2) {
    if (threadIdx.x < width)
      maxima[threadIdx.x] = fmax(maxima[threadIdx.x], maxima[threadIdx.x + width]);
    __syncthreads();
  }
  if (threadIdx.x == 0) shifts[qi] = maxima[0];
}
__global__ void screen_tensor(SummaryView s, const BlockEnvelope* boxes, int Q, double* out) {
  size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= size_t(Q) * s.h) return;
  auto x = screen_row(s, boxes + (p / s.h) * s.B * s.h, p % s.h);
  double* y = out + p * 7;
  y[0] = x.candidate;
  y[1] = x.rounded;
  y[2] = x.lower;
  y[3] = x.upper;
  y[4] = x.lower_residual;
  y[5] = x.upper_residual;
  y[6] = x.numerical_accept;
}
__device__ inline double warp_sum(double x) {
  for (int offset = 16; offset; offset /= 2) x += __shfl_down_sync(0xffffffff, x, offset);
  return x;
}
// Four warps cooperate on each output coordinate. Summation order changes, so
// this is a numerical scheduling ablation and is never used for imported boxes.
__global__ void screen_tensor_parallel(SummaryView s, const BlockEnvelope* boxes, int Q,
                                       double* out) {
  int row = blockIdx.x, j = row % s.h, qi = row / s.h;
  if (qi >= Q) return;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __shared__ double totals[3][4], candidate, rounded, lower, upper, mass;
  __shared__ int valid_warp[4], valid_row;
  double den = 0, num = 0, zlo = 0;
  bool valid = true;
  for (int b = threadIdx.x; b < s.B; b += blockDim.x) {
    BlockEnvelope e = boxes[(size_t(qi) * s.B + b) * s.h + j];
    double nu = s.nu[b * s.h + j];
    den += e.zhat;
    num += nu * e.zhat + e.mhat;
    zlo += e.zlo;
    valid = valid && finite(e.zlo) && finite(e.zhi) && finite(e.mlo) && finite(e.mhi) &&
            finite(e.zhat) && finite(e.mhat) && finite(nu) && e.zlo >= 0 && e.zlo <= e.zhi &&
            e.mlo <= e.mhi && e.zhat > 0;
  }
  den = warp_sum(den);
  num = warp_sum(num);
  zlo = warp_sum(zlo);
  int all_valid = __all_sync(0xffffffff, valid);
  if (lane == 0) {
    totals[0][warp] = den;
    totals[1][warp] = num;
    totals[2][warp] = zlo;
    valid_warp[warp] = all_valid;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    den = 0;
    num = 0;
    mass = 0;
    valid_row = 1;
    for (int w = 0; w < 4; ++w) {
      den += totals[0][w];
      num += totals[1][w];
      mass += totals[2][w];
      valid_row &= valid_warp[w];
    }
    candidate = num / den;
    rounded = bf16_nearest(candidate);
    lower = NAN;
    upper = NAN;
    if (finite(rounded)) bf16_cell(rounded, lower, upper);
    valid_row = valid_row && finite(candidate) && finite(rounded) && mass > 0;
  }
  __syncthreads();
  double rl = 0, ru = 0;
  for (int b = threadIdx.x; b < s.B; b += blockDim.x) {
    BlockEnvelope e = boxes[(size_t(qi) * s.B + b) * s.h + j];
    double nu = s.nu[b * s.h + j];
    double a = nu - lower, z = nu - upper;
    rl += e.mlo + fmin(a * e.zlo, a * e.zhi);
    ru += e.mhi + fmax(z * e.zlo, z * e.zhi);
  }
  rl = warp_sum(rl);
  ru = warp_sum(ru);
  if (lane == 0) {
    totals[0][warp] = rl;
    totals[1][warp] = ru;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    rl = 0;
    ru = 0;
    for (int w = 0; w < 4; ++w) {
      rl += totals[0][w];
      ru += totals[1][w];
    }
    double* y = out + size_t(row) * 7;
    y[0] = candidate;
    y[1] = rounded;
    y[2] = lower;
    y[3] = upper;
    y[4] = rl;
    y[5] = ru;
    y[6] = valid_row && finite(rl) && finite(ru) && rl > 0 && ru < 0;
  }
}
}  // namespace cmk
static bool valid_view(const cmk::SummaryView* view, int Q) {
  if (!view || view->B <= 0 || view->d <= 0 || view->h <= 0 || view->r < 0 || view->r > view->d ||
      Q <= 0)
    return false;
  const size_t limit = 2147483647;
  if (size_t(Q) * view->B > limit || size_t(Q) * view->h > limit || size_t(Q) * view->d > limit ||
      size_t(view->B) * view->d > limit || size_t(view->B) * view->h > limit ||
      size_t(view->B) * view->r > limit || size_t(view->r) * view->r > limit ||
      size_t(view->B) * view->r * view->r > limit || size_t(view->B) * view->r * view->h > limit)
    return false;
  return view->count && view->mu && view->nu && view->eta && view->kr && view->vr &&
         (!view->r || (view->cov && view->cross && view->diagonal));
}
extern "C" int cmk_run(const cmk::SummaryView* view, const double* q, int Q, double* shifts,
                       cmk::BlockEnvelope* boxes, double* out, void* stream_ptr, int shared) {
  if (!valid_view(view, Q) || !q || !shifts || !boxes || !out || shared < 0 || shared > 2)
    return int(cudaErrorInvalidValue);
  auto s = *view;
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  cmk::query_shifts<<<Q, 128, 0, stream>>>(s, q, Q, shifts);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) return int(err);
  if (shared)
    cmk::evaluate_blocks_shared<<<Q * s.B, 128, 0, stream>>>(s, q, shifts, Q, boxes);
  else
    cmk::evaluate_blocks<<<(size_t(Q) * s.B * s.h + 127) / 128, 128, 0, stream>>>(s, q, shifts, Q,
                                                                                  boxes);
  err = cudaGetLastError();
  if (err != cudaSuccess) return int(err);
  if (shared == 2)
    cmk::screen_tensor_parallel<<<Q * s.h, 128, 0, stream>>>(s, boxes, Q, out);
  else
    cmk::screen_tensor<<<(size_t(Q) * s.h + 127) / 128, 128, 0, stream>>>(s, boxes, Q, out);
  return int(cudaGetLastError());
}
// Timing-only entry point: callers first execute cmk_run to initialize buffers.
extern "C" int cmk_phase(const cmk::SummaryView* view, const double* q, int Q, double* shifts,
                         cmk::BlockEnvelope* boxes, double* out, void* stream_ptr, int phase) {
  if (!valid_view(view, Q) || !q || !shifts || !boxes || !out || phase < 0 || phase > 4)
    return int(cudaErrorInvalidValue);
  auto s = *view;
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (phase == 0) cmk::query_shifts<<<Q, 128, 0, stream>>>(s, q, Q, shifts);
  if (phase == 1) cmk::evaluate_blocks_shared<<<Q * s.B, 128, 0, stream>>>(s, q, shifts, Q, boxes);
  if (phase == 2)
    cmk::screen_tensor<<<(size_t(Q) * s.h + 127) / 128, 128, 0, stream>>>(s, boxes, Q, out);
  if (phase == 3)
    cmk::evaluate_blocks<<<(size_t(Q) * s.B * s.h + 127) / 128, 128, 0, stream>>>(s, q, shifts, Q,
                                                                                  boxes);
  if (phase == 4) cmk::screen_tensor_parallel<<<Q * s.h, 128, 0, stream>>>(s, boxes, Q, out);
  return int(cudaGetLastError());
}
