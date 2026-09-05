// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "imported.hpp"
namespace cmk {
// Conditional checker: ALL imported endpoints must already be sound enclosures.
// This is intentionally separate from the non-outward numerical summary path.

__device__ inline double product_lower(double al,double au,double bl,double bu){
 return fmin(fmin(__dmul_rd(al,bl),__dmul_rd(al,bu)),
             fmin(__dmul_rd(au,bl),__dmul_rd(au,bu)));
}
__device__ inline double product_upper(double al,double au,double bl,double bu){
 return fmax(fmax(__dmul_ru(al,bl),__dmul_ru(al,bu)),
             fmax(__dmul_ru(au,bl),__dmul_ru(au,bu)));
}
}
#endif
