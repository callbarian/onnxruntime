// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/framework/tuning_results.h"

namespace onnxruntime {

class IExecutionProvider;
class TuningResultsManager;
class TuningResultsValidator;

class ITuningContext {
 public:
  virtual ~ITuningContext() = default;

  virtual void EnableTunableOp() = 0;
  virtual void DisableTunableOp() = 0;
  virtual bool IsTunableOpEnabled() const = 0;

  virtual TuningResultsManager& GetTuningResultsManager() = 0;
  virtual const TuningResultsManager& GetTuningResultsManager() const = 0;

  // For validating tuning results .
  virtual const TuningResultsValidator& GetTuningResultsValidator() const = 0;
};

class TuningResultsManager {
 public:
  TuningResultsManager() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TuningResultsManager);

  KernelMap Lookup(const std::string& op_signature) const;
  int Lookup(const std::string& op_signature, const std::string& params_signature) const;

  void Add(const std::string& op_signature, const std::string& params_signature, int best_id);

  void Load(const std::unordered_map<std::string, KernelMap>& results_to_load);
  std::unordered_map<std::string, KernelMap> Dump() const;

  void Merge(const std::string& op_signature, const KernelMap& kernel_map);

 private:
  mutable OrtMutex lock_;
  std::unordered_map<std::string, KernelMap> results_;
};

class TuningResultsValidator {
 public:
  using CheckFunc = std::function<Status(const std::string&)>;
  using WriteFunc = std::function<std::string()>;
  using CheckWriteFuncs = std::unordered_map<std::string, std::pair<CheckFunc, WriteFunc>>;

  TuningResultsValidator();

  Status CheckAll(const std::unordered_map<std::string, std::string>& to_check) const;
  std::unordered_map<std::string, std::string> WriteAll() const;

 protected:
  void RegisterValidator(const std::string& key, const CheckFunc& cf, const WriteFunc& wf);

  virtual Status CheckOrtVersion(const std::string& value) const;
  virtual std::string WriteOrtVersion() const;

  virtual Status CheckOrtGitCommit(const std::string& value) const;
  virtual std::string WriteOrtGitCommit() const;

  virtual Status CheckOrtBuildConfig(const std::string& value) const;
  virtual std::string WriteOrtBuildConfig() const;

 private:
  CheckWriteFuncs validators_;
};

}  // namespace onnxruntime
