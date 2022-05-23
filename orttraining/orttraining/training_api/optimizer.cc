// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_provider.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/include/session.h"
#include "orttraining/training_api/include/utils.h"
#include "orttraining/training_api/include/optimizer.h"

namespace onnxruntime {
namespace training {
namespace api {

Optimizer::Optimizer(const std::string& optim_path_or_bytes,
                     const std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters) {
  parameters_ = parameters;
  std::unordered_map<std::string, ParameterOptimizerState>&
      param_named_optimizer_states = optimizer_state_.param_named_optimizer_states;

  // TODO: share threadpool with module session
  std::unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(nullptr, env) == Status::OK(), "Enviroment creation fails.");
  optim_sess_ = std::move(std::make_unique<InferenceSession>(GetSessionOptions(), *env));

  std::unordered_map<std::string, std::shared_ptr<IExecutionProvider>>& execution_providers = GetRegisteredExecutionProviders();
  for (auto& execution_provider : execution_providers) {
    ORT_THROW_IF_ERROR(optim_sess_->RegisterExecutionProvider(execution_provider.second));
  }
  ORT_THROW_IF_ERROR(optim_sess_->Load(optim_path_or_bytes));
  ORT_THROW_IF_ERROR(optim_sess_->Initialize());
  auto& optim_sess_state = optim_sess_->GetSessionState();

  for (auto& pair : parameters) {
    if (pair.second->RequiresGrad()) {
      param_named_optimizer_states.insert({pair.first, ParameterOptimizerState()});
      ParameterOptimizerState& cur_param_optimizer_states = param_named_optimizer_states[pair.first];
      for (auto& state_name : MOMENT_STATE_NAMES) {
        OrtValue param_state;
        // TODO: should reset the state to zero (for both CPU or CUDA Tensors.)
        ORT_ENFORCE(OrtValueLike(optim_sess_state, pair.second->Data(), param_state).IsOK());
        cur_param_optimizer_states.momentum_named_states.insert({state_name, std::move(param_state)});
      }
    }
  }

  std::shared_ptr<onnxruntime::Model> model;
  ORT_THROW_IF_ERROR(onnxruntime::Model::Load(optim_path_or_bytes, model, nullptr, env->GetLoggingManager()->DefaultLogger()));
  GetGraphInputOutputNames(model->MainGraph(), input_names_, output_names_);
  ORT_ENFORCE(input_names_[0] == "learning_rate");  // TODO: make this better
  ORT_ENFORCE(input_names_[1] == "step");           // TODO: make this better

  std::string param_name;
  std::vector<std::string> param_names, grad_names, moment1_names, moment2_names, user_inputs;
  for (size_t i = 2; i < input_names_.size(); i++) {
    std::string& name = input_names_[i];
    auto it = parameters_.find(name);
    if (it != parameters_.end()) {  // is param
      param_names.push_back(name);
      inputs_.push_back(it->second->Data());
    } else if (GetParamNameFromGradient(name, param_name)) {
      grad_names.emplace_back(name);
      // assert param_name is valid.
      auto it = parameters_.find(param_name);
      ORT_ENFORCE(it != parameters_.end(), "Unknown param: ", param_name, " for field: ", name);
      inputs_.push_back(it->second->Gradient());
    } else if (GetParamNameFromSuffix(name, MOMENT_1_SUFFIX, param_name)) {
      moment1_names.push_back(name);
      auto it = parameters_.find(param_name);
      ORT_ENFORCE(it != parameters_.end(), "Unknown param: ", param_name, " for field: ", name);
      inputs_.push_back(param_named_optimizer_states.at(param_name).momentum_named_states.at(MOMENT_STATE_NAMES[0]));
    } else if (GetParamNameFromSuffix(name, MOMENT_2_SUFFIX, param_name)) {
      moment2_names.push_back(name);
      auto it = parameters_.find(param_name);
      ORT_ENFORCE(it != parameters_.end(), "Unknown param: ", param_name, " for field: ", name);
      inputs_.push_back(param_named_optimizer_states.at(param_name).momentum_named_states.at(MOMENT_STATE_NAMES[1]));
    } else {
      ORT_THROW("This is an invalid graph. Optimizer graph contains unknown user input:", name);
    }
  }
}

Status Optimizer::Step() {
  OrtValue learning_rate_input, step_input;
  WarpInOrtValue<float>(optimizer_state_.learning_rate, &learning_rate_input);
  WarpInOrtValue<int64_t>(optimizer_state_.step, &step_input);
  std::vector<OrtValue> feeds({learning_rate_input, step_input});
  feeds.insert(feeds.end(), inputs_.begin(), inputs_.end());

  std::vector<OrtValue> outputs;
  auto status = optim_sess_->Run(RunOptions(), input_names_, feeds, output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);

  // extract step output and update
  // TODO: need to remove hardcoding
  optimizer_state_.step = GetValue<int64_t>(outputs[0]);

  return status;
}

Status Optimizer::GetStateDict(OptimizerCheckpointState& optimizer_checkpoint_state) {
  auto& grouped_optimizer_states = optimizer_checkpoint_state.group_named_optimizer_states;

  // Currently all parameters are in a single group, so we hardcode group0 here.
  // To support multiple groups, Optimizer constructor need accept informations for groupping.
  const std::string group_zero_name = "group0";
  grouped_optimizer_states.insert({group_zero_name, std::make_shared<GroupOptimizerState>(optimizer_state_)});

  // Pass the optimizer session data transfer manager for data copying when saving.
  // An alternative is, we can do copy at this stage.
  ORT_RETURN_IF_NOT(optim_sess_, "optimizer session not initialized");
  const DataTransferManager& sess_data_transfer_manager = optim_sess_->GetDataTransferManager();
  optimizer_checkpoint_state.optimizer_session_data_transfer_mgr = &sess_data_transfer_manager;
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime