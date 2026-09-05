// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
namespace cmk {
struct ImportedBox {
  double zlo, zhi, mlo, mhi, centerlo, centerhi;
};
#ifndef __CUDACC__
// Conservative one-ULP expansion of each basic binary64 operation.
// Assumes ordinary IEEE-754 binary64 evaluation, no fast math or contraction.
// The source-to-machine correspondence has not been verified in Lean.
static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8);
inline double down(double x) {
  return std::nextafter(x, -INFINITY);
}
inline double up(double x) {
  return std::nextafter(x, INFINITY);
}
inline double prod_down(double al, double au, double bl, double bu) {
  return std::min({down(al * bl), down(al * bu), down(au * bl), down(au * bu)});
}
inline double prod_up(double al, double au, double bl, double bu) {
  return std::max({up(al * bl), up(al * bu), up(au * bl), up(au * bu)});
}
inline bool check_imported(const ImportedBox* boxes, int B, double lower, double upper) {
  bool valid = std::isfinite(lower) && std::isfinite(upper) && lower < upper;
  double rl = 0, ru = 0, zl = 0;
  for (int b = 0; b < B; ++b) {
    const auto x = boxes[b];
    valid = valid && std::isfinite(x.zlo) && std::isfinite(x.zhi) && std::isfinite(x.mlo) &&
            std::isfinite(x.mhi) && std::isfinite(x.centerlo) && std::isfinite(x.centerhi) &&
            x.zlo >= 0 && x.zlo <= x.zhi && x.mlo <= x.mhi && x.centerlo <= x.centerhi;
    double al = down(x.centerlo - lower), au = up(x.centerhi - lower);
    double bl = down(x.centerlo - upper), bu = up(x.centerhi - upper);
    valid =
        valid && std::isfinite(al) && std::isfinite(au) && std::isfinite(bl) && std::isfinite(bu);
    if (!valid) return false;
    rl = down(rl + down(x.mlo + prod_down(al, au, x.zlo, x.zhi)));
    ru = up(ru + up(x.mhi + prod_up(bl, bu, x.zlo, x.zhi)));
    zl = down(zl + x.zlo);
  }
  return valid && std::isfinite(rl) && std::isfinite(ru) && zl > 0 && rl > 0 && ru < 0;
}
#endif
}  // namespace cmk
