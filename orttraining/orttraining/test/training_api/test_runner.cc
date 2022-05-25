// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>
#include "orttraining/training_api/include/interfaces.h"

#include "cxxopts.hpp"
#include "synthetic_data_loader.h"

using namespace onnxruntime::training::api;
struct TestRunnerParameters {
  // Models configs.
  std::string model_training_graph_path;
  std::optional<std::string> model_evaluation_graph_path;
  std::string optimizer_training_graph_path;
  std::string checkpoint_to_load_path;
  std::string model_name;

  // Data configs.
  std::string train_data_dir;
  std::string test_data_dir;
  std::string output_dir;  // Output of training, e.g., trained model files.

  // Training configs.
  size_t train_batch_size;
  size_t num_train_epochs;
  size_t eval_batch_size;
  size_t eval_interval;
  size_t checkpoint_interval;
  int gradient_accumulation_steps = 1;
};

void EnforceCheck(bool run_ret, std::string err_msg) {
  if (!run_ret) {
    throw std::runtime_error("EnforceCheck failed: " + err_msg);
  }
}

bool ParseArguments(int argc, char* argv[], TestRunnerParameters& params) {
  cxxopts::Options options("Training API Test", "Main Program to test training C++ APIs.");
  // clang-format off
  options
    .add_options()
      ("model_training_graph_path", "The path to the training model to load. ",
        cxxopts::value<std::string>()->default_value(""))
      ("model_evaluation_graph_path", "The path to the evaluation model to load. ",
        cxxopts::value<std::string>()->default_value(""))
      ("optimizer_training_graph_path", "The path to the optimizer graph to load. ",
        cxxopts::value<std::string>()->default_value(""))
      ("checkpoint_to_load_path", "The path to the checkpoint to load if provided.",
        cxxopts::value<std::string>()->default_value(""))
      ("model_name", "The name of the model.",
        cxxopts::value<std::string>()->default_value("model_test"))

      ("train_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/train"))
      ("test_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/test"))
      ("output_dir", "The output directory where the trained model files will be written.",
        cxxopts::value<std::string>()->default_value(""))

      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>())
      ("eval_batch_size", "Total batch size for eval.", cxxopts::value<int>())
      ("num_train_epochs", "Total number of training epochs to perform.", cxxopts::value<int>()->default_value("100"))
      ("eval_interval", "Number of training steps before doing evaluation.", cxxopts::value<int>()->default_value("1000"))
      ("checkpoint_interval", "Number of training steps before saving checkpoint.", cxxopts::value<int>()->default_value("1000"))
      ("gradient_accumulation_steps", "The number of gradient accumulation steps before performing a backward/update pass.",
        cxxopts::value<int>()->default_value("1"));

  // clang-format on

  try {
    auto flags = options.parse(argc, argv);

    params.model_training_graph_path = flags["model_training_graph_path"].as<std::string>();
    std::string eval_path = flags["model_evaluation_graph_path"].as<std::string>();
    if (eval_path.empty()) {
      params.model_evaluation_graph_path = std::nullopt;
    } else {
      params.model_evaluation_graph_path = eval_path;
    }

    params.optimizer_training_graph_path = flags["optimizer_training_graph_path"].as<std::string>();
    params.checkpoint_to_load_path = flags["checkpoint_to_load_path"].as<std::string>();
    params.model_name = flags["model_name"].as<std::string>();

    params.train_batch_size = flags["train_batch_size"].as<int>();
    if (flags.count("eval_batch_size")) {
      params.eval_batch_size = flags["eval_batch_size"].as<int>();
    } else {
      params.eval_batch_size = params.train_batch_size;
    }
    params.num_train_epochs = flags["num_train_epochs"].as<int>();
    params.eval_interval = flags["eval_interval"].as<int>();
    params.checkpoint_interval = flags["checkpoint_interval"].as<int>();

    params.gradient_accumulation_steps = flags["gradient_accumulation_steps"].as<int>();
    EnforceCheck(params.gradient_accumulation_steps >= 1,
                 "Invalid gradient_accumulation_steps parameter: should be >= 1");

    params.train_data_dir = flags["train_data_dir"].as<std::string>();
    params.test_data_dir = flags["test_data_dir"].as<std::string>();
    params.output_dir = flags["output_dir"].as<std::string>();
    if (params.output_dir.empty()) {
      printf("No output directory specified. Trained model files will not be saved.\n");
    }

  } catch (const std::exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    std::cerr << msg << ": " << e.what() << "\n"
              << options.help() << "\n";
    return false;
  }

  return true;
}

void RunTraining(const TestRunnerParameters& params) {
  const auto& api = Ort::GetApi();

  CheckpointState state;
  // TODO: update using public API's calling pattern, e.g. api.LoadCheckpoint().
  EnforceCheck(LoadCheckpoint(params.checkpoint_to_load_path, state).IsOK(), "Failed to load checkpoint");

  OrtSessionOptions* session_options = nullptr;
  EnforceCheck(api.CreateSessionOptions(&session_options) == nullptr, "Failed to create session options.");

#ifdef USE_CUDA
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  EnforceCheck(api.CreateCUDAProviderOptions(&cuda_options) == nullptr, "Failed to create cuda provider options");
  EnforceCheck(api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options) == nullptr,
               "Failed to append cuda ep.");
#endif

  OrtEnv* env = nullptr;
  EnforceCheck(api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "e2e_test_runner", &env) == nullptr, "Failed to create env");

  // TODO: update using public API's calling pattern, e.g. api.CreateModule().
  Ort::OrtModule module(api, env, session_options,
                        params.model_training_graph_path,
                        state.module_checkpoint_state.named_parameters,
                        params.model_evaluation_graph_path);

  bool do_eval = params.model_evaluation_graph_path.has_value();

  // TODO: update using public API's calling pattern, e.g. api.CreateOptimizer().
  Ort::OrtOptimizer optimizer(api, env, session_options,
                              params.optimizer_training_graph_path,
                              state.module_checkpoint_state.named_parameters);

  size_t sample_count_per_epoch = 4;
  onnxruntime::training::test::training_api::SyntheticDataLoader data_loader(sample_count_per_epoch, params.train_batch_size);

  int64_t total_step_count = static_cast<int64_t>(params.num_train_epochs * data_loader.NumOfBatches());
  int64_t warmup_step_count = total_step_count / 3;

  // TODO: update using public API's calling pattern, e.g. api.CreateLinearLRScheduler().
  Ort::OrtLinearLRScheduler scheduler = Ort::OrtLinearLRScheduler(optimizer, warmup_step_count, total_step_count);

  const size_t stabilized_perf_start_step = 0;
  double stabilized_total_end_to_end_time{0};
  auto end_to_end_start = std::chrono::high_resolution_clock::now();

  for (size_t epoch = 0, batch_idx = 0; epoch < params.num_train_epochs; ++epoch) {
    // for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
    if (batch_idx >= stabilized_perf_start_step) {
      end_to_end_start = std::chrono::high_resolution_clock::now();
    }

    std::vector<Ort::Value> inputs;
    data_loader.GetNextBatch(inputs);

    std::vector<Ort::Value> fetches;
    EnforceCheck(module.TrainStep(inputs, fetches), "Failed during module.TrainStep.");

    float loss = *(fetches[0].GetTensorMutableData<float>());
    std::cout << "Batch # : " << batch_idx << " Loss: " << loss << std::endl;

    if ((batch_idx + 1) % params.gradient_accumulation_steps == 0) {
      // Gradient accumulation steps completed.
      EnforceCheck(optimizer.Step(), "Failed during optimizer.Step().");
      // Update learning rate.
      EnforceCheck(scheduler.Step(), "Failed during shceduler.Step()");
      EnforceCheck(module.ResetGrad(), "Failed during module.ResetGrad().");
    }

    if (do_eval && (batch_idx + 1) % params.eval_interval == 0) {
      std::vector<Ort::Value> eval_results;
      EnforceCheck(module.EvalStep(inputs, eval_results), "Failed during Module.EvalStep().");
    }

    if ((batch_idx + 1) % params.checkpoint_interval == 0) {
      // Save trained weights
      CheckpointState state_to_save;
      EnforceCheck(module.GetStateDict(state_to_save.module_checkpoint_state), "Failed to load module states.");
      EnforceCheck(optimizer.GetStateDict(state_to_save.optimizer_checkpoint_state), "Failed to load optimizer states.");
      state_to_save.property_bag.AddProperty<int64_t>(std::string("epoch"), static_cast<int64_t>(epoch));
      std::string ckpt_file = params.output_dir + "/ckpt_" + params.model_name + std::to_string(batch_idx);

      // TODO: update using public API's calling pattern, e.g. api.SaveCheckpoint().
      EnforceCheck(SaveCheckpoint(state_to_save, ckpt_file).IsOK(), "Failed to save checkpoint.");
    }

    batch_idx++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_seconds = end - end_to_end_start;
  stabilized_total_end_to_end_time = duration_seconds.count();

  std::cout << "Training completed - end to end latency: " << stabilized_total_end_to_end_time << "(s)" << std::endl;

  api.ReleaseEnv(env);

#ifdef USE_CUDA
  // Finally, don't forget to release the provider options
  api.ReleaseCUDAProviderOptions(cuda_options);
#endif

  api.ReleaseSessionOptions(session_options);
}

int main(int argc, char* argv[]) {
  TestRunnerParameters params;
  EnforceCheck(ParseArguments(argc, argv, params), "Parse arguments failed.");

  // Start training session
  RunTraining(params);
  return 0;
}