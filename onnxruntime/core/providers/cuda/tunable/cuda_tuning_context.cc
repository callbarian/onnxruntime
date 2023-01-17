// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tunable/cuda_tunable.h"

#include "core/framework/tuning_context.h"
#define TUNING_CONTEXT_IMPL
#include "core/framework/tuning_context_impl.h"
#undef TUNING_CONTEXT_IMPL

namespace onnxruntime {
namespace cuda {
namespace tunable {

std::string WriteCudaVersion() {
  int version;
  CUDA_CALL_THROW(cudaRuntimeGetVersion(&version));
  return std::to_string(version);
}

Status CheckCudaVersion(const std::string& value) {
  auto current = WriteCudaVersion();
  ORT_RETURN_IF(current != value, "CUDA runtime version mismatch: tuning results produced with CUDA ", value,
                ", onnxruntime currently run with CUDA ", current);
  return Status::OK();
}

CudaTuningResultsValidator::CudaTuningResultsValidator() {
  RegisterValidator("CUDA_VERSION", CheckCudaVersion, WriteCudaVersion);
}

CudaTuningContext::CudaTuningContext(TunableOpInfo* info) : info_(info) {}

void CudaTuningContext::EnableTunableOp() {
  LOGS_DEFAULT(INFO) << "Enable TunableOp for CUDA Execution Provider";
  info_->enabled = true;
}

void CudaTuningContext::DisableTunableOp() {
  LOGS_DEFAULT(INFO) << "Disable TunableOp for CUDA Execution Provider";
  info_->enabled = false;
}

bool CudaTuningContext::IsTunableOpEnabled() const {
  return info_->enabled;
}

TuningResultsManager& CudaTuningContext::GetTuningResultsManager() {
  return manager_;
}

const TuningResultsManager& CudaTuningContext::GetTuningResultsManager() const {
  return manager_;
}

const TuningResultsValidator& CudaTuningContext::GetTuningResultsValidator() const {
  return validator_;
}

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
