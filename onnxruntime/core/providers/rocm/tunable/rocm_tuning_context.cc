// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/tunable/rocm_tunable.h"

#include "core/framework/tuning_context.h"
#define TUNING_CONTEXT_IMPL
#include "core/framework/tuning_context_impl.h"
#undef TUNING_CONTEXT_IMPL

namespace onnxruntime {
namespace rocm {
namespace tunable {

// std::string WriteRocmVersion() {
//   return std::to_string(ROCM_VERSION);
// }

// Status CheckRocmVersion(const std::string& value) {
//   auto current = WriteRocmVersion();
//   ORT_RETURN_IF(current != value, "ROCm version mismatch: tuning results produced with ROCm ", value,
//                 ", current library built with ", current);
//   return Status::OK();
// }

std::string WriteHipVersion() {
  int version;
  HIP_CALL_THROW(hipRuntimeGetVersion(&version));
  return std::to_string(version);
}

Status CheckHipVersion(const std::string& value) {
  auto current = WriteHipVersion();
  ORT_RETURN_IF(current != value, "HIP runtime version mismatch: tuning results produced with HIP ", value,
                ", onnxruntime currently run with HIP ", current);
  return Status::OK();
}

std::string WriteRocBlasVersion() {
  char buf[64];
  ROCBLAS_CALL_THROW(rocblas_get_version_string(buf, 256));
  buf[63] = '\0';
  return buf;
}

Status CheckRocBlasVersion(const std::string& value) {
  auto current = WriteRocBlasVersion();
  ORT_RETURN_IF(current != value, "rocblas runtime version mismatch: tuning results produced with rocblas ", value,
                ", onnxruntime currently run with rocblas ", current);
  return Status::OK();
}

RocmTuningResultsValidator::RocmTuningResultsValidator() {
  RegisterValidator("HIP_VERSION", CheckHipVersion, WriteHipVersion);
  RegisterValidator("ROCBLAS_VERSION", CheckRocBlasVersion, WriteRocBlasVersion);
}

std::string RocmTuningResultsValidator::WriteOrtBuildConfig() const {
  std::ostringstream oss;
  oss << "USE_CK=" << USE_COMPOSABLE_KERNEL << "|";
#ifdef USE_ROCBLAS_EXTENSION_API
  oss << "USE_ROCBLAS_EXTENSION_API" << 1 << "|";
#else
  oss << "USE_ROCBLAS_EXTENSION_API" << 0 << "|";
#endif
  return oss.str();
}

RocmTuningContext::RocmTuningContext(TunableOpInfo* info) : info_(info) {}

void RocmTuningContext::EnableTunableOp() {
  LOGS_DEFAULT(INFO) << "Enable TunableOp for ROCm Execution Provider";
  info_->enabled = true;
}

void RocmTuningContext::DisableTunableOp() {
  LOGS_DEFAULT(INFO) << "Disable TunableOp for ROCm Execution Provider";
  info_->enabled = false;
}

bool RocmTuningContext::IsTunableOpEnabled() const {
  return info_->enabled;
}

TuningResultsManager& RocmTuningContext::GetTuningResultsManager() {
  return manager_;
}

const TuningResultsManager& RocmTuningContext::GetTuningResultsManager() const {
  return manager_;
}

const TuningResultsValidator& RocmTuningContext::GetTuningResultsValidator() const {
  return validator_;
}

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
