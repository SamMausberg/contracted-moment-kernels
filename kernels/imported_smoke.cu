// SPDX-License-Identifier: Apache-2.0
// Replays outward rational exports and deliberate rejection controls on CUDA.
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "moments.cu"

static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Allocations {
  std::vector<void*> ptrs;
  ~Allocations() {
    for (auto p : ptrs) cudaFree(p);
  }
  template <class T>
  T* upload(const std::vector<T>& x) {
    T* p = nullptr;
    check(cudaMalloc((void**)&p, x.size() * sizeof(T)));
    ptrs.push_back(p);
    check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }
};
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: cmk_cuda_imported fixture.txt");
    std::ifstream f(argv[1]);
    int rows = 0, B = 0;
    f >> rows >> B;
    if (!f || rows < 1 || B < 1 || size_t(rows) * B > 10000000)
      throw std::runtime_error("Invalid dimensions");
    std::vector<cmk::ImportedBox> boxes;
    std::vector<double> lower, upper;
    std::vector<unsigned char> expected;
    for (int row = 0; row < rows; ++row) {
      double l = 0, u = 0;
      int e = 0;
      f >> l >> u >> e;
      lower.push_back(l);
      upper.push_back(u);
      expected.push_back(e);
      for (int b = 0; b < B; ++b) {
        cmk::ImportedBox x;
        f >> x.zlo >> x.zhi >> x.mlo >> x.mhi >> x.centerlo >> x.centerhi;
        boxes.push_back(x);
      }
      if (!f || e < 0 || e > 1) throw std::runtime_error("Malformed fixture");
    }
    f >> std::ws;
    if (!f.eof()) throw std::runtime_error("Unexpected fixture suffix");
    // The control box represents y=1; (-1,2) strictly contains it.
    // Every mutation below must abstain, even if other coordinates look benign.
    constexpr int controls = 13;
    for (int c = 0; c < controls; ++c) {
      double l = -1, u = 2;
      cmk::ImportedBox x{1, 1, 0, 0, 1, 1};
      if (c == 0) x.zlo = -1;
      if (c == 1) x.zlo = 2;
      if (c == 2) x.mlo = 1;
      if (c == 3) x.centerlo = 2;
      if (c == 4) x.zlo = x.zhi = 0;
      if (c == 5) x.mhi = std::numeric_limits<double>::infinity();
      if (c == 6) x.centerlo = std::numeric_limits<double>::quiet_NaN();
      if (c == 7) l = u;
      if (c == 8) l = std::numeric_limits<double>::quiet_NaN();
      if (c == 9) l = 1;  // strict midpoint/boundary equality
      if (c == 10) u = 1;
      if (c == 11) {
        x.centerlo = x.centerhi = DBL_MAX;
        l = -DBL_MAX;
        u = DBL_MAX;
      }
      if (c == 12) {
        x.zlo = x.zhi = DBL_MAX;
        x.centerlo = x.centerhi = DBL_MAX;
      }
      lower.push_back(l);
      upper.push_back(u);
      expected.push_back(0);
      for (int b = 0; b < B; ++b) boxes.push_back(x);
    }
    int total = rows + controls;
    std::vector<unsigned char> actual(total, 0);
    Allocations a;
    auto db = a.upload(boxes);
    auto dl = a.upload(lower);
    auto du = a.upload(upper);
    auto dp = a.upload(actual);
    cmk::screen_imported_boxes<<<(total + 127) / 128, 128>>>(db, dl, du, total, B, dp);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(actual.data(), dp, total, cudaMemcpyDeviceToHost));
    int accepted = 0, oracle = 0;
    for (int row = 0; row < total; ++row) {
      if (actual[row] && !expected[row])
        throw std::runtime_error("False directed certificate/control acceptance");
      accepted += actual[row];
      oracle += expected[row];
    }
    std::cout << "{\"gpu_imported_pass\":true,\"rational_rows\":" << rows
              << ",\"rational_certified_rows\":" << oracle << ",\"gpu_certified_rows\":" << accepted
              << ",\"rejection_controls\":" << controls << ",\"false_certificates\":0}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
