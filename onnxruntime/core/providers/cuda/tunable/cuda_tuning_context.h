// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tuning_context.h"

#include "core/providers/cuda/cuda_execution_provider_info.h"

namespace onnxruntime {
namespace cuda {
namespace tunable {

class CudaTuningContext : public ITuningContext {
 public:
  explicit CudaTuningContext(TunableOpInfo* info);

  void EnableTunableOp() override;
  void DisableTunableOp() override;
  bool IsTunableOpEnabled() const override;

  TuningResultsManager& GetTuningResultsManager() override;
  const TuningResultsManager& GetTuningResultsManager() const override;

 private:
  TunableOpInfo* info_;  // non-owning handle
  TuningResultsManager manager_;
};

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
