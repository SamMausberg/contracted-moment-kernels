// SPDX-License-Identifier: Apache-2.0
#include "fixture.hpp"
#include "moments.cu"
#include <iostream>
#include <algorithm>
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Allocations{
 std::vector<void*> ptrs;
 ~Allocations(){for(void*p:ptrs)cudaFree(p);}
 template<class T>T* allocate(size_t n){T*p=nullptr;if(n){check(cudaMalloc((void**)&p,n*sizeof(T)));ptrs.push_back(p);}return p;}
 template<class T>T* upload(const std::vector<T>&v){auto*p=allocate<T>(v.size());if(v.size())check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
};
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: cmk_cuda_smoke fixture.bin");
 cmk::Fixture f(argv[1]);Allocations a;auto s=f.view();
 s.count=a.upload(f.count);s.mu=a.upload(f.mu);s.nu=a.upload(f.nu);s.cov=a.upload(f.cov);
 s.cross=a.upload(f.cross);s.diagonal=a.upload(f.diagonal);s.eta=a.upload(f.eta);s.kr=a.upload(f.kr);s.vr=a.upload(f.vr);
 auto*q=a.upload(f.queries);auto*shift=a.upload(f.shifts);
 size_t nb=cmk::product({f.Q,f.B,f.h}),nr=cmk::product({f.Q,f.h});
 auto*boxes=a.allocate<cmk::BlockEnvelope>(nb);auto*out=a.allocate<cmk::ScreenResult>(nr);
 cmk::evaluate_blocks<<<(nb+127)/128,128>>>(s,q,shift,f.Q,boxes);check(cudaGetLastError());
 cmk::reduce_and_screen<<<(nr+127)/128,128>>>(s,boxes,f.Q,out);check(cudaGetLastError());check(cudaDeviceSynchronize());
 std::vector<cmk::BlockEnvelope> hb(nb);std::vector<cmk::ScreenResult> ho(nr);
 check(cudaMemcpy(hb.data(),boxes,nb*sizeof(hb[0]),cudaMemcpyDeviceToHost));
 check(cudaMemcpy(ho.data(),out,nr*sizeof(ho[0]),cudaMemcpyDeviceToHost));
 double worst=0;int accepted=0;
 for(size_t p=0;p<nb;++p){auto&e=hb[p];double x[]={e.zlo,e.zhi,e.mlo,e.mhi,e.zhat,e.mhat};
  for(int k=0;k<6;++k){double y=f.expected_boxes[p*6+k];
   if(!cmk::finite(x[k])||!cmk::finite(y))throw std::runtime_error("nonfinite envelope on fixture");
   worst=std::max(worst,std::abs(x[k]-y)/(1+std::abs(y)));}}
 for(size_t p=0;p<nr;++p){auto&x=ho[p];
  if(x.numerical_accept!=f.expected_gate[p])throw std::runtime_error("CUDA numerical screen mismatch");
  if(x.numerical_accept && !(x.lower<f.expected_y[p]&&f.expected_y[p]<x.upper))
  throw std::runtime_error("false numerical screen on fixture");accepted+=x.numerical_accept;}
 if(worst>1e-9)throw std::runtime_error("CUDA numerical mismatch");
 std::cout<<"{\"gpu_smoke_pass\":true,\"max_scaled_difference\":"<<worst
          <<",\"numerical_pass_coordinates\":"<<accepted<<"}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
