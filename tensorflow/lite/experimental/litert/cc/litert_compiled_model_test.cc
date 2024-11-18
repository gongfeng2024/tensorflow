// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"

#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"

constexpr const float kTestInput0Tensor[] = {1, 2};
constexpr const float kTestInput1Tensor[] = {10, 20};
constexpr const float kTestOutputTensor[] = {11, 22};
constexpr const size_t kTestOutputSize =
    sizeof(kTestOutputTensor) / sizeof(kTestOutputTensor[0]);

namespace litert {
namespace {

static constexpr absl::string_view kTfliteFile =
    "third_party/tensorflow/lite/experimental/litert/test/testdata/"
    "simple_model.tflite";

TEST(CompiledModelTest, Basic) {
  auto res_compiled_model = CompiledModel::CreateFromTflFile(kTfliteFile);
  ASSERT_TRUE(res_compiled_model) << "Failed to initialize CompiledModel";
  auto& compiled_model = **res_compiled_model;

  auto signatures = compiled_model.GetSignatures();
  EXPECT_EQ(signatures.size(), 1);
  EXPECT_EQ(signatures[0], CompiledModel::kDefaultSignatureKey);

  auto input_buffers_res = compiled_model.CreateInputBuffers(signatures[0]);
  EXPECT_TRUE(input_buffers_res.HasValue());
  std::vector<TensorBuffer>& input_buffers = *input_buffers_res;

  auto output_buffers_res = compiled_model.CreateOutputBuffers(signatures[0]);
  EXPECT_TRUE(output_buffers_res.HasValue());
  std::vector<TensorBuffer>& output_buffers = *output_buffers_res;

  // Fill model inputs.
  auto input_names = compiled_model.GetInputNames(signatures[0]);
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_STREQ(input_names[0], "arg0");
  EXPECT_STREQ(input_names[1], "arg1");
  auto& input_0_buffer = input_buffers[0];
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(input_0_buffer);
    ASSERT_TRUE(lock_and_addr.HasValue());
    std::memcpy(lock_and_addr->second, kTestInput0Tensor,
                sizeof(kTestInput0Tensor));
  }
  auto& input_1_buffer = input_buffers[1];
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(input_1_buffer);
    ASSERT_TRUE(lock_and_addr.HasValue());
    std::memcpy(lock_and_addr->second, kTestInput1Tensor,
                sizeof(kTestInput1Tensor));
  }

  // Execute model.
  compiled_model.Invoke(signatures[0], input_buffers, output_buffers);

  // Check model output.
  auto output_names = compiled_model.GetOutputNames(signatures[0]);
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_STREQ(output_names[0], "tfl.add");
  auto& output_buffer = output_buffers[0];
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create(output_buffer);
    ASSERT_TRUE(lock_and_addr.HasValue());
    const float* output = reinterpret_cast<const float*>(lock_and_addr->second);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    for (auto i = 0; i < kTestOutputSize; ++i) {
      EXPECT_NEAR(output[i], kTestOutputTensor[i], 1e-5);
    }
  }
}

}  // namespace
}  // namespace litert
