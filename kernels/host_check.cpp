// SPDX-License-Identifier: Apache-2.0
#include "fixture.hpp"
#include <iostream>
#include <algorithm>
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: cmk_host_check fixture.bin");
 cmk::Fixture f(argv[1]);auto s=f.view();double worst=0;int accepted=0;
 std::vector<cmk::BlockEnvelope> boxes(cmk::product({f.Q,f.B,f.h}));
 for(int q=0;q<f.Q;++q)for(int b=0;b<f.B;++b)for(int j=0;j<f.h;++j){
  size_t p=(q*f.B+b)*f.h+j;auto e=cmk::evaluate_block(s,f.queries.data()+q*f.d,b,j,f.shifts[q]);boxes[p]=e;
  double xs[]={e.zlo,e.zhi,e.mlo,e.mhi,e.zhat,e.mhat};
  for(int k=0;k<6;++k){double want=f.expected_boxes[p*6+k];
   if(!cmk::finite(xs[k])||!cmk::finite(want))throw std::runtime_error("nonfinite envelope on fixture");
   worst=std::max(worst,std::abs(xs[k]-want)/(1+std::abs(want)));}
 }
 for(int q=0;q<f.Q;++q)for(int j=0;j<f.h;++j){
  auto x=cmk::screen_row(s,boxes.data()+q*f.B*f.h,j);
  if(x.numerical_accept!=f.expected_gate[q*f.h+j])throw std::runtime_error("screen mismatch");
  double y=f.expected_y[q*f.h+j];
  if(x.numerical_accept && !(x.lower<y&&y<x.upper))throw std::runtime_error("false numerical screen");
  accepted+=x.numerical_accept;
 }
 if(worst>1e-11)throw std::runtime_error("envelope mismatch");
 std::cout<<"{\"host_core_pass\":true,\"max_scaled_difference\":"<<worst
          <<",\"numerical_pass_coordinates\":"<<accepted<<",\"coordinates\":"<<f.Q*f.h<<"}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
