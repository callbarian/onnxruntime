// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tuning_context.h"

#include "core/providers/rocm/rocm_execution_provider_info.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

class RocmTuningResultsValidator : public TuningResultsValidator {
 public:
  RocmTuningResultsValidator();

 protected:
  std::string WriteOrtBuildConfig() const override;
};

class RocmTuningContext : public ITuningContext {
 public:
  explicit RocmTuningContext(TunableOpInfo* info);

  void EnableTunableOp() override;
  void DisableTunableOp() override;
  bool IsTunableOpEnabled() const override;

  TuningResultsManager& GetTuningResultsManager() override;
  const TuningResultsManager& GetTuningResultsManager() const override;

  const TuningResultsValidator& GetTuningResultsValidator() const override;

 private:
  TunableOpInfo* info_;  // non-owning handle
  TuningResultsManager manager_;
  RocmTuningResultsValidator validator_;
};

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
