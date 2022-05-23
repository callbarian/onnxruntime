// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/inference_session.h"
#include <onnxruntime_cxx_api.h>

namespace onnxruntime {
namespace training {
namespace api {

struct Parameter {
 public:
  // Create parameter
  Parameter(std::string name, const OrtValue& data)
      : name_(name), data_(data) {
    ORT_ENFORCE(data_.IsAllocated());
  }

  // Return the mutable data.
  OrtValue& Data() { return data_; }
  std::string Name() const { return name_; }

  // Return if trainable. The trainable property of a param
  // cannot change over the lifetime of the on-device training
  // session since the gradient graph is prebuilt for this setting.
  bool RequiresGrad() const { return requires_grad_; }

  // Return the mutable gradient for trainable parameter.
  OrtValue& Gradient() { return gradient_; }
  std::string GradientName() const { return gradient_name_; }

  // Reset and release the gradient buffer of this Parameter.
  Status ResetGrad();

  Status SetRequiresGrad(bool requires_grad) {
    requires_grad_ = requires_grad;
    return Status::OK();
  }

 protected:
  Status AllocateGrad(const std::string& gradient_name, const SessionState& allocator);

  // need to set grad but not public api
 private:
  std::string name_;
  OrtValue data_;

  OrtValue gradient_;
  std::string gradient_name_;

  // Whether the param is trainable. The optimizer state is
  // only created for a trainable param
  bool requires_grad_{true};
  friend class Module;
};

struct ModuleCheckpointState {
 public:
  std::unordered_map<std::string, std::shared_ptr<Parameter>> named_parameters;
  const DataTransferManager* train_session_data_transfer_mgr;
};

struct Module {
 public:
  // Initialize a module from an ORT inference session with loaded
  // training ONNX model and load parameters
  Module(const std::string& train_model_path_or_bytes,
         std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
         const std::optional<std::string>& eval_model_path_or_bytes = std::nullopt);

  // Return the trainable/nontrainable parameters
  std::vector<std::shared_ptr<Parameter>> parameters() const;

  std::unordered_map<std::string, std::shared_ptr<Parameter>> named_parameters() const {
    return parameters_;
  }

  // Reset and release the gradient buffer of all trainable params
  Status ResetGrad();

  // Train Step – does forward and backward computation. The outputs will be the forward’s outputs.
  // Gradients will be accumulated within the Parameter object
  Status TrainStep(const std::vector<Ort::Value>& /*inputs*/, std::vector<Ort::Value>& /*outputs*/);

  // Eval Step – does forward computation. This will use a separate inference session
  // and take in a separate inference graph, while sharing the parameters
  Status EvalStep(const std::vector<Ort::Value>& /*inputs*/, std::vector<Ort::Value>& /*outputs*/);

  // Return the states of the module as a map.
  Status GetStateDict(ModuleCheckpointState& module_checkpoint_states);

 private:
  Status TrainStepInternal(const std::vector<OrtValue>& /*inputs*/, std::vector<OrtValue>& /*outputs*/);

  Status EvalStepInternal(const std::vector<OrtValue>& /*inputs*/, std::vector<OrtValue>& /*outputs*/);

  std::unique_ptr<onnxruntime::InferenceSession> train_sess_;
  std::unique_ptr<onnxruntime::InferenceSession> eval_sess_;
  std::unordered_map<std::string, std::shared_ptr<Parameter>> parameters_;
  std::vector<std::string> train_input_names_;
  std::vector<std::string> train_output_names_;
  std::vector<std::string> eval_input_names_;
  std::vector<std::string> eval_output_names_;
  std::vector<OrtValue> weights_;
  std::vector<OrtValue> gradients_;
  bool lazy_reset_grad_ = false;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime