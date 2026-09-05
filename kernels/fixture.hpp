// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "moments.cuh"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <cstring>
#include <initializer_list>
namespace cmk {
inline size_t product(std::initializer_list<int> dims) {
  size_t n=1;
  for(int d:dims){if(d<0 || (d && n>100000000/size_t(d)))throw std::runtime_error("fixture too large");n*=d;}
  return n;
}
template<class T>std::vector<T> readvec(std::ifstream& in,size_t n){
  std::vector<T> v(n);if(n)in.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));
  if(!in)throw std::runtime_error("truncated fixture");return v;
}
struct Fixture {
 int n,B,d,r,h,Q;
 std::vector<int32_t> offsets,expected_gate;
 std::vector<double> keys,values,count,mu,nu,cov,cross,diagonal,eta,kr,vr,queries,shifts,expected_boxes,expected_y;
 explicit Fixture(const std::string& path){
  static_assert(sizeof(double)==8 && sizeof(int32_t)==4,"unsupported host layout");
  uint16_t endian=1;if(*reinterpret_cast<unsigned char*>(&endian)!=1)throw std::runtime_error("little endian required");
  std::ifstream in(path,std::ios::binary);char magic[8];in.read(magic,8);
  if(!in || std::memcmp(magic,"CMK0001\n",8))throw std::runtime_error("invalid fixture magic");
  auto dims=readvec<int32_t>(in,6);n=dims[0];B=dims[1];d=dims[2];r=dims[3];h=dims[4];Q=dims[5];
  if(n<=0||n>100000000||B<=0||B>n||d<=0||d>100000||r<0||r>d||h<=0||h>100000||Q<=0||Q>100000)throw std::runtime_error("invalid dimensions");
  offsets=readvec<int32_t>(in,product({B+1}));
  if(offsets.front()!=0||offsets.back()!=n)throw std::runtime_error("invalid partition");
  for(int b=0;b<B;++b)if(offsets[b]>=offsets[b+1])throw std::runtime_error("invalid partition");
  keys=readvec<double>(in,product({n,d}));values=readvec<double>(in,product({n,h}));
  count=readvec<double>(in,B);mu=readvec<double>(in,product({B,d}));nu=readvec<double>(in,product({B,h}));
  cov=readvec<double>(in,product({B,r,r}));cross=readvec<double>(in,product({B,r,h}));
  diagonal=readvec<double>(in,product({B,r,h}));eta=readvec<double>(in,product({B,h}));
  kr=readvec<double>(in,product({B,d}));vr=readvec<double>(in,product({B,h}));
  queries=readvec<double>(in,product({Q,d}));shifts=readvec<double>(in,Q);
  expected_boxes=readvec<double>(in,product({Q,B,h,6}));expected_y=readvec<double>(in,product({Q,h}));
  expected_gate=readvec<int32_t>(in,product({Q,h}));
  if(in.peek()!=EOF)throw std::runtime_error("unexpected fixture suffix");
 }
 SummaryView view()const{return {B,d,r,h,count.data(),mu.data(),nu.data(),cov.data(),cross.data(),
  diagonal.data(),eta.data(),kr.data(),vr.data()};}
};
}
