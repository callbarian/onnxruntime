// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "core/providers/cuda/tunable/util.h"
#elif USE_ROCM
#include <hip/hip_runtime.h>
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "core/providers/rocm/tunable/util.h"
#endif

#ifdef USE_CUDA
using onnxruntime::cuda::TunableOpInfo;
using onnxruntime::cuda::tunable::Timer;
using TuningContextT = onnxruntime::cuda::tunable::CudaTuningContext;
using StreamT = cudaStream_t;
#elif USE_ROCM
using onnxruntime::rocm::TunableOpInfo;
using onnxruntime::rocm::tunable::Timer;
using TuningContextT = onnxruntime::rocm::tunable::RocmTuningContext;
using StreamT = hipStream_t;
#endif

/// Wrapping around Op and TunableOp
class IKernelExplorer {
 public:
  virtual void Run() = 0;

  void SetRepeats(int n) {
    repeats_ = n;
  }

  float Profile() {
    // warm up
    for (int i = 0; i < 5; i++) {
      Run();
    }
    Timer timer{Stream()};
    timer.Start();
    for (int i = 0; i < repeats_; i++) {
      Run();
    }
    timer.End();
    return timer.Duration() / repeats_;
  }

  virtual ~IKernelExplorer() = default;

 protected:
  TuningContextT* TuningContext() {
    if (tuning_ctx_ == nullptr) {
#if USE_CUDA || USE_ROCM
      tuning_ctx_ = std::make_unique<TuningContextT>(&info_);
#else
      ORT_NOT_IMPLEMENTED("only CUDA or ROCM is supported");
#endif
    }

    return tuning_ctx_.get();
  }

  StreamT Stream() { return stream_; }

 private:
  TunableOpInfo info_{};
  std::unique_ptr<TuningContextT> tuning_ctx_{};
  StreamT stream_{0};
  int repeats_{100};
};
