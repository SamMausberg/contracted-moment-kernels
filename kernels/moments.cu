// SPDX-License-Identifier: Apache-2.0
// Draft CUDA kernels. No GH200 compilation or performance claim is attached.
#include "moments.cuh"
#include "directed.cuh"
#include <cuda_runtime.h>
namespace cmk {
__global__ void evaluate_blocks(SummaryView s,const double* q,const double* shift,
                               int Q,BlockEnvelope* out){
 size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(p>=size_t(Q)*s.B*s.h)return;
 int j=p%s.h,b=(p/s.h)%s.B,qi=p/(s.B*s.h);
 out[p]=evaluate_block(s,q+qi*s.d,b,j,shift[qi]);
}
__global__ void reduce_and_screen(SummaryView s,const BlockEnvelope* in,int Q,ScreenResult* out){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=Q*s.h)return;
 out[p]=screen_row(s,in+(p/s.h)*s.B*s.h,p%s.h);
}
// Sparse dense-block correction primitive. A device queue/controller is not supplied.
// Values are numerical approximations, not outward intervals. The keys remain live.
__global__ void scan_selected_blocks(SummaryView s,const double* q,const double* shift,
 const double* K,const double* V,const int* offsets,const int* selected,int Q,BlockEnvelope* out){
 size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p>=size_t(Q)*s.B*s.h)return;
 int j=p%s.h,b=(p/s.h)%s.B,qi=p/(s.B*s.h);if(!selected[qi*s.B+b])return;
 double z=0,m=0;
 for(int i=offsets[b];i<offsets[b+1];++i){
  double score=0;for(int a=0;a<s.d;++a)score+=q[qi*s.d+a]*K[i*s.d+a];
  double w=exp(score-shift[qi]);z+=w;m+=w*(V[i*s.h+j]-s.nu[b*s.h+j]);
 }
 out[p]={z,z,m,m,z,m};
}
// Actual directed-rounding boundary evaluation, conditional on sound imported boxes.
// CPU-created rational enclosures may be exported outward. Approximate boxes from
// evaluate_blocks MUST NOT be used here as though that supplied the missing premise.
__global__ void screen_imported_boxes(const ImportedBox* boxes,const double* lower,
 const double* upper,int rows,int B,unsigned char* pass){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 double rl=0,ru=0,zl=0;bool valid=finite(lower[row])&&finite(upper[row])&&lower[row]<upper[row];
 for(int b=0;b<B;++b){
  ImportedBox x=boxes[row*B+b];
  valid=valid&&finite(x.zlo)&&finite(x.zhi)&&finite(x.mlo)&&finite(x.mhi)
    &&finite(x.centerlo)&&finite(x.centerhi)&&x.zlo>=0&&x.zlo<=x.zhi
    &&x.mlo<=x.mhi&&x.centerlo<=x.centerhi;
  double al=__dsub_rd(x.centerlo,lower[row]),au=__dsub_ru(x.centerhi,lower[row]);
  double bl=__dsub_rd(x.centerlo,upper[row]),bu=__dsub_ru(x.centerhi,upper[row]);
  valid=valid&&finite(al)&&finite(au)&&finite(bl)&&finite(bu);
  rl=__dadd_rd(rl,__dadd_rd(x.mlo,product_lower(al,au,x.zlo,x.zhi)));
  ru=__dadd_ru(ru,__dadd_ru(x.mhi,product_upper(bl,bu,x.zlo,x.zhi)));
  zl=__dadd_rd(zl,x.zlo);
 }
 pass[row]=valid&&finite(rl)&&finite(ru)&&zl>0&&rl>0&&ru<0;
}
}
