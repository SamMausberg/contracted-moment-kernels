// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cfloat>
#include <cmath>
#ifdef __CUDACC__
#define CMK_HD __host__ __device__
#else
#define CMK_HD
#endif
namespace cmk {
// Numeric summary evaluation only. These doubles are NOT outward enclosures.
struct SummaryView {
  int B, d, r, h;
  const double *count, *mu, *nu, *cov, *cross, *diagonal, *eta, *kr, *vr;
};
struct BlockEnvelope {
  double zlo, zhi, mlo, mhi, zhat, mhat;
};
// Query/block quantities shared across value channels by the CUDA ablation.
// Numerical approximations only; no outward-rounding claim is made.
struct BlockScalar {
  double w, tau, q2, pu, zlo, zhi, zhat, discarded;
};
struct ScreenResult {
  double candidate, rounded, lower, upper, lower_residual, upper_residual;
  int numerical_accept;
};
CMK_HD inline bool finite(double x) {
  return x == x && fabs(x) <= DBL_MAX;
}
CMK_HD inline double sharp_tail(double x) {
  if (!finite(x) || x < 0 || x > 650) return HUGE_VAL;
  if (x == 0) return 0;
  if (x < .5) {
    double term = x / 6, total = term;
    for (int k = 4; k < 36; ++k) {
      term *= x / k;
      total += term;
    }
    return total;
  }
  return (expm1(x) - x - x * x / 2) / (x * x);
}
CMK_HD inline BlockEnvelope evaluate_block(SummaryView s, const double* q, int b, int j,
                                           double shift) {
  double score = 0, rho = 0, eps = 0, variance = 0, q2 = 0, c = 0;
  for (int a = 0; a < s.d; ++a) {
    score += q[a] * s.mu[b * s.d + a];
    if (a < s.r)
      rho += fabs(q[a]) * s.kr[b * s.d + a];
    else
      eps += fabs(q[a]) * s.kr[b * s.d + a];
  }
  for (int a = 0; a < s.r; ++a) {
    q2 += q[a] * q[a];
    c += q[a] * s.cross[(b * s.r + a) * s.h + j] +
         q[a] * q[a] * s.diagonal[(b * s.r + a) * s.h + j] / 2;
    for (int k = 0; k < s.r; ++k) variance += q[a] * q[k] * s.cov[(b * s.r + a) * s.r + k];
  }
  variance = fmax(variance, 0.0);
  double w = s.count[b] * exp(score - shift), tau = sharp_tail(rho) * variance;
  double A = 1 + variance / 2, pl = w * fmax(1., A - tau), pu = w * (A + tau);
  double mh = w * c, R = s.vr[b * s.h + j];
  double beta = w * (s.eta[b * s.h + j] * q2 / 2 + tau * R) + expm1(eps) * pu * R;
  return {pl / exp(eps), pu * exp(eps), mh - beta, mh + beta, w * A, mh};
}
CMK_HD inline BlockScalar evaluate_scalar(SummaryView s, const double* q, int b, double shift) {
  double score = 0, rho = 0, eps = 0, variance = 0, q2 = 0;
  for (int a = 0; a < s.d; ++a) {
    score += q[a] * s.mu[b * s.d + a];
    if (a < s.r)
      rho += fabs(q[a]) * s.kr[b * s.d + a];
    else
      eps += fabs(q[a]) * s.kr[b * s.d + a];
  }
  for (int a = 0; a < s.r; ++a) {
    q2 += q[a] * q[a];
    for (int k = 0; k < s.r; ++k) variance += q[a] * q[k] * s.cov[(b * s.r + a) * s.r + k];
  }
  variance = fmax(variance, 0.0);
  double w = s.count[b] * exp(score - shift), tau = sharp_tail(rho) * variance;
  double A = 1 + variance / 2, pl = w * fmax(1., A - tau), pu = w * (A + tau);
  return {w, tau, q2, pu, pl / exp(eps), pu * exp(eps), w * A, expm1(eps)};
}
CMK_HD inline BlockEnvelope evaluate_channel(SummaryView s, const double* q, int b, int j,
                                             BlockScalar x) {
  double c = 0;
  for (int a = 0; a < s.r; ++a)
    c += q[a] * s.cross[(b * s.r + a) * s.h + j] +
         q[a] * q[a] * s.diagonal[(b * s.r + a) * s.h + j] / 2;
  double mh = x.w * c, R = s.vr[b * s.h + j];
  double beta = x.w * (s.eta[b * s.h + j] * x.q2 / 2 + x.tau * R) + x.discarded * x.pu * R;
  return {x.zlo, x.zhi, mh - beta, mh + beta, x.zhat, mh};
}
CMK_HD inline double bf16_nearest(double x) {
  if (!finite(x) || fabs(x) > ldexp(1., 120)) return NAN;
  if (x == 0) return 0;
  double a = fabs(x);
  int exponent;
  frexp(a, &exponent);
  --exponent;
  int p = exponent - 7;
  if (p < -133) p = -133;
  double step = ldexp(1., p), z = a / step, whole = floor(z), fraction = z - whole;
  if (fraction > .5 || (fraction == .5 && fmod(whole, 2.) != 0)) whole += 1;
  return copysign(whole * step, x);
}
CMK_HD inline void bf16_cell(double b, double& lo, double& hi) {
  if (b == 0) {
    lo = -ldexp(1., -134);
    hi = ldexp(1., -134);
    return;
  }
  double a = fabs(b);
  int e;
  frexp(a, &e);
  --e;
  int p = e - 7;
  if (p < -133) p = -133;
  double step = ldexp(1., p);
  double prev = (a == ldexp(1., e) && e > -126) ? step / 2 : step;
  double l = a - prev / 2, u = a + step / 2;
  if (b < 0) {
    lo = -u;
    hi = -l;
  } else {
    lo = l;
    hi = u;
  }
}
CMK_HD inline ScreenResult screen_row(SummaryView s, const BlockEnvelope* boxes, int j) {
  double den = 0, num = 0, masslo = 0;
  bool ok = true;
  for (int b = 0; b < s.B; ++b) {
    BlockEnvelope e = boxes[b * s.h + j];
    double nu = s.nu[b * s.h + j];
    den += e.zhat;
    num += nu * e.zhat + e.mhat;
    masslo += e.zlo;
    ok = ok && finite(e.zlo) && finite(e.zhi) && finite(e.mlo) && finite(e.mhi) && finite(e.zhat) &&
         finite(e.mhat) && finite(nu) && e.zlo >= 0 && e.zlo <= e.zhi && e.mlo <= e.mhi &&
         e.zhat > 0;
  }
  double candidate = num / den, rounded = bf16_nearest(candidate), lo = NAN, hi = NAN;
  if (finite(rounded)) bf16_cell(rounded, lo, hi);
  double rl = 0, ru = 0;
  for (int b = 0; b < s.B; ++b) {
    BlockEnvelope e = boxes[b * s.h + j];
    double nu = s.nu[b * s.h + j];
    double a = nu - lo, z = nu - hi;
    rl += e.mlo + fmin(a * e.zlo, a * e.zhi);
    ru += e.mhi + fmax(z * e.zlo, z * e.zhi);
  }
  return {candidate,
          rounded,
          lo,
          hi,
          rl,
          ru,
          int(ok && masslo > 0 && finite(candidate) && finite(rounded) && rl > 0 && ru < 0)};
}
}  // namespace cmk
