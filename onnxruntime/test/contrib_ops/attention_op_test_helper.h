// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <vector>
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
using contrib::AttentionMaskType;

namespace test {

#ifndef ENABLE_TRAINING  // TRT fused attention is enabled only on non-training builds

// Return packed weights and bias for input projection.
void GetAttentionWeight(std::vector<float>& weight_data, int elements = 64 * 3 * 64, int offset = 0, int step=1);
void GetAttentionBias(std::vector<float>& bias_data, int elements = 3 * 64, int offset = 0, int step=1);

struct AttentionTestData{
    int hidden_size;
    int v_hidden_size;
    int num_heads;
    int batch_size;
    int sequence_length;
    int kv_sequence_length;
    AttentionMaskType mask_type;
    std::vector<int> key_padding_mask_data;
    std::vector<float> query_data;
    std::vector<float> key_data;
    std::vector<float> value_data;
    std::vector<float> bias_data;
    std::vector<float> output_data;
};

void GetCrossAttentionData(AttentionTestData& data);
#endif

}  // namespace test
}  // namespace onnxruntime
