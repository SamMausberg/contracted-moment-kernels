// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <iostream>

#include "benchmark_api.cu"
#include "fixture.hpp"
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Allocations {
  std::vector<void*> ptrs;
  ~Allocations() {
    for (void* p : ptrs) cudaFree(p);
  }
  template <class T>
  T* allocate(size_t n) {
    T* p = nullptr;
    if (n) {
      check(cudaMalloc((void**)&p, n * sizeof(T)));
      ptrs.push_back(p);
    }
    return p;
  }
  template <class T>
  T* upload(const std::vector<T>& v) {
    auto* p = allocate<T>(v.size());
    if (v.size()) check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }
};
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: cmk_cuda_smoke fixture.bin");
    cmk::Fixture f(argv[1]);
    Allocations a;
    auto s = f.view();
    s.count = a.upload(f.count);
    s.mu = a.upload(f.mu);
    s.nu = a.upload(f.nu);
    s.cov = a.upload(f.cov);
    s.cross = a.upload(f.cross);
    s.diagonal = a.upload(f.diagonal);
    s.eta = a.upload(f.eta);
    s.kr = a.upload(f.kr);
    s.vr = a.upload(f.vr);
    auto* q = a.upload(f.queries);
    auto* shift = a.upload(f.shifts);
    size_t nb = cmk::product({f.Q, f.B, f.h}), nr = cmk::product({f.Q, f.h});
    auto* boxes = a.allocate<cmk::BlockEnvelope>(nb);
    auto* out = a.allocate<cmk::ScreenResult>(nr);
    cmk::evaluate_blocks<<<(nb + 127) / 128, 128>>>(s, q, shift, f.Q, boxes);
    check(cudaGetLastError());
    cmk::reduce_and_screen<<<(nr + 127) / 128, 128>>>(s, boxes, f.Q, out);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    std::vector<cmk::BlockEnvelope> hb(nb);
    std::vector<cmk::ScreenResult> ho(nr);
    check(cudaMemcpy(hb.data(), boxes, nb * sizeof(hb[0]), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(ho.data(), out, nr * sizeof(ho[0]), cudaMemcpyDeviceToHost));
    double worst = 0;
    int accepted = 0;
    for (size_t p = 0; p < nb; ++p) {
      auto& e = hb[p];
      double x[] = {e.zlo, e.zhi, e.mlo, e.mhi, e.zhat, e.mhat};
      for (int k = 0; k < 6; ++k) {
        double y = f.expected_boxes[p * 6 + k];
        if (!cmk::finite(x[k]) || !cmk::finite(y))
          throw std::runtime_error("nonfinite envelope on fixture");
        worst = std::max(worst, std::abs(x[k] - y) / (1 + std::abs(y)));
      }
    }
    for (size_t p = 0; p < nr; ++p) {
      auto& x = ho[p];
      if (x.numerical_accept != f.expected_gate[p])
        throw std::runtime_error("CUDA numerical screen mismatch");
      if (x.numerical_accept && !(x.lower < f.expected_y[p] && f.expected_y[p] < x.upper))
        throw std::runtime_error("false numerical screen on fixture");
      accepted += x.numerical_accept;
    }
    if (worst > 1e-9) throw std::runtime_error("CUDA numerical mismatch");
    auto original = hb;
    cmk::evaluate_blocks_shared<<<f.Q * f.B, 128>>>(s, q, shift, f.Q, boxes);
    check(cudaGetLastError());
    cmk::reduce_and_screen<<<(nr + 127) / 128, 128>>>(s, boxes, f.Q, out);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(hb.data(), boxes, nb * sizeof(hb[0]), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(ho.data(), out, nr * sizeof(ho[0]), cudaMemcpyDeviceToHost));
    double shared_worst = 0;
    for (size_t p = 0; p < nb; ++p) {
      auto& e = hb[p];
      auto& o = original[p];
      double x[] = {e.zlo, e.zhi, e.mlo, e.mhi, e.zhat, e.mhat};
      double y[] = {o.zlo, o.zhi, o.mlo, o.mhi, o.zhat, o.mhat};
      for (int k = 0; k < 6; ++k) {
        if (!cmk::finite(x[k])) throw std::runtime_error("Nonfinite shared evaluator output");
        shared_worst = std::max(shared_worst, std::abs(x[k] - y[k]) / (1 + std::abs(y[k])));
      }
    }
    for (size_t p = 0; p < nr; ++p)
      if (ho[p].numerical_accept != f.expected_gate[p])
        throw std::runtime_error("Shared screen mismatch");
    if (shared_worst > 1e-12) throw std::runtime_error("Shared evaluator mismatch");
    auto* parallel = a.allocate<double>(nr * 7);
    for (int control = 0; control < 4; ++control) {
      auto invalid = s;
      if (control == 0) invalid.B = 0;
      if (control == 1) invalid.r = invalid.d + 1;
      if (control == 2) invalid.B = 2147483647;
      const double* query = control == 3 ? nullptr : q;
      if (cmk_run(&invalid, query, f.Q, shift, boxes, parallel, nullptr, 2) !=
          int(cudaErrorInvalidValue))
        throw std::runtime_error("Invalid benchmark ABI input was not rejected");
    }
    check(static_cast<cudaError_t>(cmk_run(&s, q, f.Q, shift, boxes, parallel, nullptr, 2)));
    check(cudaDeviceSynchronize());
    std::vector<double> hp(nr * 7);
    check(cudaMemcpy(hp.data(), parallel, nr * 7 * sizeof(double), cudaMemcpyDeviceToHost));
    double parallel_worst = 0;
    for (size_t p = 0; p < nr; ++p) {
      if (int(hp[p * 7 + 6]) != f.expected_gate[p])
        throw std::runtime_error("Parallel screen mismatch");
      parallel_worst = std::max(parallel_worst, std::abs(hp[p * 7] - ho[p].candidate));
    }
    if (parallel_worst > 1e-12) throw std::runtime_error("Parallel candidate mismatch");
    auto* keys = a.upload(f.keys);
    auto* values = a.upload(f.values);
    auto* offsets = a.upload(f.offsets);
    auto* selected = a.upload(std::vector<int>(f.Q * f.B, 1));
    cmk::scan_selected_blocks<<<(nb + 127) / 128, 128>>>(s, q, shift, keys, values, offsets,
                                                         selected, f.Q, boxes);
    check(cudaGetLastError());
    cmk::reduce_and_screen<<<(nr + 127) / 128, 128>>>(s, boxes, f.Q, out);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(ho.data(), out, nr * sizeof(ho[0]), cudaMemcpyDeviceToHost));
    double correction_worst = 0;
    for (size_t p = 0; p < nr; ++p) {
      if (!cmk::finite(ho[p].candidate)) throw std::runtime_error("Nonfinite correction output");
      correction_worst = std::max(correction_worst, std::abs(ho[p].candidate - f.expected_y[p]));
    }
    if (correction_worst > 1e-11) throw std::runtime_error("Dense block correction mismatch");
    std::cout << "{\"gpu_smoke_pass\":true,\"max_scaled_difference\":" << worst
              << ",\"numerical_pass_coordinates\":" << accepted
              << ",\"shared_max_scaled_difference\":" << shared_worst
              << ",\"parallel_candidate_max_difference\":" << parallel_worst
              << ",\"abi_rejection_controls\":4"
              << ",\"all_block_correction_max_absolute_error\":" << correction_worst << "}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
