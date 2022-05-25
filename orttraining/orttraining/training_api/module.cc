// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"
#include "orttraining/training_api/include/module.h"
#include "orttraining/training_api/include/utils.h"

#include <onnxruntime_cxx_api.h>

using namespace onnxruntime;

namespace onnxruntime {
namespace training {
namespace api {

Status Parameter::AllocateGrad(const std::string& gradient_name, const SessionState& sess_state) {
  // assert param is allocated
  ORT_ENFORCE(data_.IsAllocated());
  ORT_ENFORCE(requires_grad_);
  gradient_name_ = gradient_name;
  ORT_ENFORCE(OrtValueLike(sess_state, data_, gradient_).IsOK());
  return Status::OK();
}

Status Parameter::ResetGrad() {
  // TODO: make use of lazy_reset_grad input instead
  Tensor* p_tensor = gradient_.GetMutable<Tensor>();
  const auto& device = p_tensor->Location().device;
  if (device.Type() == OrtDevice::CPU) {
    memset(p_tensor->MutableDataRaw(), 0, p_tensor->SizeInBytes());
  }
#if defined(USE_CUDA) || defined(USE_ROCM)
  else if (device.Type() == OrtDevice::GPU) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
  }
#endif
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown device type ", device.Type(), " for param:", name_);
  }
  return Status::OK();
}

Module::Module(std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
               InferenceSession* train_session) {
  // create value copy as same params are passed to Optimizer constructor
  parameters_ = parameters;

  train_sess_ = train_session;
  auto& train_sess_state = train_sess_->GetSessionState();
  GetGraphInputOutputNames(train_sess_, train_input_names_, train_output_names_);

  std::vector<std::string> param_input_names, grad_input_names, user_input_names;
  std::string param_name;
  for (auto input_name : train_input_names_) {
    auto it = parameters_.find(input_name);
    if (it != parameters_.end()) {
      param_input_names.emplace_back(input_name);
      weights_.push_back(it->second->Data());
    } else if (GetParamNameFromGradient(input_name, param_name)) {
      grad_input_names.emplace_back(input_name);
      // create gradient buffer
      // assert param_name is valid.
      auto it = parameters_.find(param_name);
      if (it != parameters_.end()) {
        ORT_THROW_IF_ERROR(it->second->AllocateGrad(input_name, train_sess_state));
        gradients_.push_back(it->second->Gradient());
      } else {
        // raise error here.
      }
    } else {
      user_input_names.emplace_back(input_name);
    }
  }
  train_input_names_ = user_input_names;
  train_input_names_.insert(train_input_names_.end(), param_input_names.begin(), param_input_names.end());
  train_input_names_.insert(train_input_names_.end(), grad_input_names.begin(), grad_input_names.end());
}

Module::Module(std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
               InferenceSession* train_session,
               InferenceSession* eval_session) : Module(parameters, train_session) {
  if (eval_session) {
    eval_sess_ = eval_session;
    GetGraphInputOutputNames(eval_sess_, eval_input_names_, eval_output_names_);
    // TODO:: do validation on eval inputs and outputs: eg order of user inputs, weights
  }
}

std::vector<std::shared_ptr<Parameter>> Module::parameters() const {
  std::vector<std::shared_ptr<Parameter>> params;
  for (auto& it : parameters_) {
    params.push_back(it.second);
  }
  return params;
}

Status Module::ResetGrad() {
  for (auto& it : parameters_) {
    ORT_ENFORCE(it.second->ResetGrad().IsOK());
  }
  return Status::OK();
}

Status Module::TrainStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs) {
  size_t output_names_len = train_output_names_.size();
  outputs.resize(output_names_len);

  std::vector<OrtValue> feeds{inputs};
  feeds.insert(feeds.end(), weights_.begin(), weights_.end());
  feeds.insert(feeds.end(), gradients_.begin(), gradients_.end());
  // TODO: consider maintaining this as ortvalue instead of bool
  OrtValue reset_grad_input;
  WarpInOrtValue<bool>(lazy_reset_grad_, &reset_grad_input);
  feeds.push_back(reset_grad_input);

  // TODO: need to filter out the grads from the output ortvalues
  auto status = train_sess_->Run(RunOptions(), train_input_names_, feeds, train_output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);
  return status;
}

Status Module::EvalStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs) {
  size_t output_names_len = eval_output_names_.size();
  outputs.resize(output_names_len);

  std::vector<OrtValue> feeds{inputs};
  feeds.insert(feeds.end(), weights_.begin(), weights_.end());
  auto status = eval_sess_->Run(RunOptions(), eval_input_names_, feeds, eval_output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);
  return status;
}

Status Module::GetStateDict(ModuleCheckpointState& module_checkpoint_state) {
  module_checkpoint_state.named_parameters = named_parameters();

  // Pass the training session data transfer manager for data copying when saving.
  // An alternative is, we can do copy at this stage.
  ORT_RETURN_IF_NOT(train_sess_, "training session not initialized");
  const DataTransferManager& sess_data_transfer_manager = train_sess_->GetDataTransferManager();
  module_checkpoint_state.train_session_data_transfer_mgr = &sess_data_transfer_manager;
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime