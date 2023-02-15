// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "performance_runner.h"
#include <google/protobuf/stubs/common.h>
#include <cuda_runtime.h>

using namespace onnxruntime;
const OrtApi* g_ort = NULL;

void RunPerformanceRunner(Ort::Env& env, perftest::PerformanceTestConfig& config, std::random_device& rd) {
  std::thread::id this_id = std::this_thread::get_id();
  perftest::PerformanceRunner perf_runner(env, config, rd);

  auto status = perf_runner.Run(this_id);
  if (!status.IsOK()) {
    printf("Run failed:%s\n", status.ErrorMessage().c_str());
    //return -1;
    std::cout << "thread id " << this_id << std::endl;
  } else {
    perf_runner.SerializeResult();
    std::cout << "Run complete on thread " << this_id << std::endl;
  }
}

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  perftest::PerformanceTestConfig test_config;
  if (!perftest::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    perftest::CommandLineParser::ShowUsage();
    return -1;
  }
  //seongmin
  //void* cuda_memory;
  //::cudaMalloc((void**)&cuda_memory, 1);
  ////::cudaFree(cuda_memory);


  //void* cuda_memory1;
  //void* cuda_memory2;
  //::cudaMalloc(&cuda_memory1, 1000);
  //cuda_memory2 = cuda_memory1;
  //cudaFree(cuda_memory1);

  //::cudaFree(cuda_memory);
  //::cudaMalloc((void**)&cuda_memory, 3000);
  //::cudaMalloc(&cuda_memory1, 150);

  //seongmin end


  Ort::Env env{nullptr};
  {
    bool failed = false;
    ORT_TRY {
      OrtLoggingLevel logging_level = test_config.run_config.f_verbose
                                          ? ORT_LOGGING_LEVEL_VERBOSE
                                          : ORT_LOGGING_LEVEL_WARNING;
      env = Ort::Env(logging_level, "Default");
    }
    ORT_CATCH(const Ort::Exception& e) {
      ORT_HANDLE_EXCEPTION([&]() {
        fprintf(stderr, "Error creating environment: %s \n", e.what());
        failed = true;
      });
    }

    if (failed)
      return -1;
  }

  std::vector<perftest::PerformanceRunner> model_vec;



  std::random_device rd;
  std::vector<std::string> model_paths;
  ////error
  //model_paths.emplace_back("E:\\source\\git_onnxruntime-1.12.1\\onnxruntime\\build\\Windows\\Debug\\Debug\\testdata\\squeezenet1.0-9\\classifier\\TimestampClassificatorResnet34.onnx");
  //model_paths.emplace_back("E:\\source\\git_onnxruntime-1.12.1\\onnxruntime\\build\\Windows\\Debug\\Debug\\testdata\\squeezenet1.0-9\\image_tagging\\recognizer\\ImgTagAttentionRec.onnx");
  //model_paths.emplace_back("E:\\source\\git_onnxruntime-1.12.1\\onnxruntime\\build\\Windows\\Debug\\Debug\\testdata\\squeezenet1.0-9\\face\\extractor\\resnet\\ArcFace_ResNet50_ms1mv3.onnx");

  ////no input
  //model_paths.emplace_back("E:\\source\\git_onnxruntime-1.12.1\\onnxruntime\\build\\Windows\\Debug\\Debug\\testdata\\squeezenet1.0-9\\face\\aligner\\Cr_MobileNet_106pt_192.onnx");

  model_paths.emplace_back("E:\\source\\git_onnxruntime-1.12.1\\onnxruntime\\build\\Windows\\Debug\\Debug\\testdata\\squeezenet1.0-9\\deepfake\\F3Net_220819.onnx");
  model_paths.emplace_back("E:\\source\\git_onnxruntime-1.12.1\\onnxruntime\\build\\Windows\\Debug\\Debug\\testdata\\squeezenet1.0-9\\face\\detector\\RetinaFace_MobileNet0.25_640.onnx");
  model_paths.emplace_back("E:\\source\\git_onnxruntime-1.12.1\\onnxruntime\\build\\Windows\\Debug\\Debug\\testdata\\squeezenet1.0-9\\face\\extractor\\mobilenet\\ArcFace_MobileFaceNet.onnx");
  model_paths.emplace_back("E:\\source\\git_onnxruntime-1.12.1\\onnxruntime\\build\\Windows\\Debug\\Debug\\testdata\\squeezenet1.0-9\\image_tagging\\detector\\ImgTagAttentionDet.onnx");
  model_paths.emplace_back("E:\\source\\git_onnxruntime-1.12.1\\onnxruntime\\build\\Windows\\Debug\\Debug\\testdata\\squeezenet1.0-9\\easyocr\\EasyOcrDetector.onnx");
  for (auto path : model_paths) {
    test_config.model_info.model_file_path = std::wstring(path.begin(), path.end());

    perftest::PerformanceRunner perf_runner1(env, test_config, rd);
    std::thread::id id;
    auto status = perf_runner1.Run(id);
    if (!status.IsOK()) {
      printf("Run failed:%s\n", status.ErrorMessage().c_str());
      return -1;
    }

    perf_runner1.SerializeResult();

    enum RUN_MODE {
      Sequential,
      Parallel
    };

    auto num_of_runs = 1;

    RUN_MODE mode = Parallel;

    if (mode == Sequential) {
      for (auto i = 0; i < num_of_runs; i++) {
        perftest::PerformanceRunner perf_runner(env, test_config, rd);
        std::thread::id id;
        auto status = perf_runner.Run(id);
        if (!status.IsOK()) {
          printf("Run failed:%s\n", status.ErrorMessage().c_str());
          return -1;
        }
        perf_runner.SerializeResult();
      }
    } else if (mode == Parallel) {
      std::vector<std::thread> workers;
      for (auto i = 0; i < num_of_runs; i++) {
        workers.push_back(std::thread(RunPerformanceRunner, std::ref(env), std::ref(test_config), std::ref(rd)));
      }

      for (auto i = 0; i < num_of_runs; i++) {
        workers[i].join();
      }
    }
  }

  return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  int retval = -1;
  ORT_TRY {
    retval = real_main(argc, argv);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      fprintf(stderr, "%s\n", ex.what());
      retval = -1;
    });
  }

  ::google::protobuf::ShutdownProtobufLibrary();

  return retval;
}
