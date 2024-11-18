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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/external_litert_buffer_context.h"
#include "tensorflow/lite/experimental/litert/runtime/tfl_utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace litert {

Expected<void> CompiledModel::Initialize() {
  // Use BuiltinOpResolverWithoutDefaultDelegates to avoid auto applying of
  // Xnnpack delegate with GetSignatureRunner() API.
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder(*fb_model_, resolver)(&interp_);
  if (interp_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  // Register the ExternalLiteRtBufferContext for TensorBuffer handshaking.
  buffer_context_ =
      std::make_unique<litert::internal::ExternalLiteRtBufferContext>();
  interp_->SetExternalContext(kTfLiteLiteRtBufferContext,
                              buffer_context_.get());

  // Construct the list of signature keys in std::string.
  auto keys = interp_->signature_keys();
  if (keys.empty()) {
    signature_keys_ = {kDefaultSignatureKey};
  } else {
    signature_keys_.reserve(keys.size());
    for (const auto& key : keys) {
      signature_keys_.push_back(*key);
    }
  }

  return Expected<void>();
}

Expected<CompiledModel::Ptr> CompiledModel::CreateFromTflFile(
    absl::string_view filename) {
  auto runtime = std::make_unique<CompiledModel>();

  auto alloc = tflite::GetAllocationFromFile(filename.data(),
                                             tflite::DefaultErrorReporter());
  if (alloc == nullptr) {
    return Unexpected(kLiteRtStatusErrorFileIO);
  }
  runtime->alloc_ = std::move(alloc);

  runtime->fb_model_ = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(runtime->alloc_->base()),
      runtime->alloc_->bytes());
  if (runtime->fb_model_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorFileIO);
  }

  runtime->Initialize();

  return runtime;
}

Expected<CompiledModel::Ptr> CompiledModel::CreateFromTflFileWithByteCode(
    absl::string_view tfl_filename, absl::string_view npu_filename) {
  auto runtime = std::make_unique<CompiledModel>();

  {
    auto model_with_byte_code =
        internal::GetModelBufWithByteCode(tfl_filename, npu_filename);
    if (!model_with_byte_code) {
      return model_with_byte_code.Error();
    }
    runtime->model_buf_ = std::move(*model_with_byte_code);
  }

  runtime->alloc_ = std::make_unique<tflite::MemoryAllocation>(
      runtime->model_buf_.Data(), runtime->model_buf_.Size(),
      tflite::DefaultErrorReporter());

  runtime->fb_model_ = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(runtime->alloc_->base()),
      runtime->alloc_->bytes());
  if (runtime->fb_model_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorFileIO);
  }

  runtime->Initialize();

  // Apply delegates. For now, DispatchDelegate is applied with the
  // CreateFromTflFileWithByteCode().
  // TODO: b/379317134 - Support other delegates with compilation options.
  auto dispatch_delegate_options = CreateDispatchDelegateOptionsPtr();
  LiteRtDispatchDelegateAddAllocBaseOption(dispatch_delegate_options.get(),
                                           runtime->alloc_->base());
  auto dispatch_delegate =
      CreateDispatchDelegatePtr(std::move(dispatch_delegate_options));
  if (auto status =
          runtime->interp_->ModifyGraphWithDelegate(dispatch_delegate.get());
      status != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to modify graph with delegate");
  }

  return runtime;
}

std::vector<const char*> CompiledModel::GetInputNames(
    absl::string_view signature_key) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return {};
  }
  return runner->input_names();
}

std::vector<const char*> CompiledModel::GetOutputNames(
    absl::string_view signature_key) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return {};
  }
  return runner->output_names();
}

litert::Expected<TensorBufferRequirements*>
CompiledModel::GetInputBufferRequirements(absl::string_view signature_key,
                                          absl::string_view input_name) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Failed to get signature runner");
  }
  auto* input_tensor = runner->input_tensor(std::string(input_name).c_str());
  if (input_tensor == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Failed to get input tensor");
  }

  auto requirements = buffer_context_->GetBufferRequirement(input_tensor);
  if (requirements.HasValue()) {
    return requirements;
  }
  auto cpu_buffer_requirements = TensorBufferRequirements::Create(
      {kLiteRtTensorBufferTypeHostMemory}, input_tensor->bytes, {0});
  if (!cpu_buffer_requirements.HasValue()) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              cpu_buffer_requirements.Error().Message());
  }
  cpu_buffer_requirements_[input_tensor] = std::move(*cpu_buffer_requirements);
  auto it = cpu_buffer_requirements_.find(input_tensor);
  return litert::Expected<TensorBufferRequirements*>(&(it->second));
}

litert::Expected<TensorBufferRequirements*>
CompiledModel::GetOutputBufferRequirements(absl::string_view signature_key,
                                           absl::string_view output_name) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Failed to get signature runner");
  }
  auto* output_tensor = runner->output_tensor(std::string(output_name).c_str());
  if (output_tensor == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Failed to get output tensor");
  }

  auto requirements = buffer_context_->GetBufferRequirement(output_tensor);
  if (requirements.HasValue()) {
    return requirements;
  }
  auto cpu_buffer_requirements = TensorBufferRequirements::Create(
      {kLiteRtTensorBufferTypeHostMemory}, output_tensor->bytes, {0});
  if (!cpu_buffer_requirements.HasValue()) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              cpu_buffer_requirements.Error().Message());
  }
  cpu_buffer_requirements_[output_tensor] = std::move(*cpu_buffer_requirements);
  auto it = cpu_buffer_requirements_.find(output_tensor);
  return litert::Expected<TensorBufferRequirements*>(&(it->second));
}

litert::Expected<RankedTensorType> CompiledModel::GetInputTensorType(
    absl::string_view signature_key, absl::string_view input_name) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Failed to get signature runner");
  }
  const auto* input_tensor =
      runner->input_tensor(std::string(input_name).c_str());
  if (input_tensor == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Failed to get input tensor");
  }
  return litert::Expected<RankedTensorType>(
      internal::ConvertTensorType(
          reinterpret_cast<const TfLiteOpaqueTensor*>(input_tensor))
          .Value());
}

litert::Expected<RankedTensorType> CompiledModel::GetOutputTensorType(
    absl::string_view signature_key, absl::string_view output_name) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Failed to get signature runner");
  }
  const auto* output_tensor =
      runner->output_tensor(std::string(output_name).c_str());
  if (output_tensor == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Failed to get output tensor");
  }
  return litert::Expected<RankedTensorType>(
      internal::ConvertTensorType(
          reinterpret_cast<const TfLiteOpaqueTensor*>(output_tensor))
          .Value());
}

litert::Expected<std::vector<TensorBuffer>> CompiledModel::CreateInputBuffers(
    absl::string_view signature_key) {
  std::vector<TensorBuffer> input_buffers;
  auto input_names = GetInputNames(signature_key);
  input_buffers.reserve(input_names.size());

  for (const auto& input_name : input_names) {
    auto input_buffer_requirements =
        GetInputBufferRequirements(signature_key, input_name);
    if (!input_buffer_requirements.HasValue()) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                input_buffer_requirements.Error().Message());
    }
    auto tensor_type = GetInputTensorType(signature_key, input_name);
    if (!tensor_type.HasValue()) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                tensor_type.Error().Message());
    }
    LiteRtTensorBufferType tensor_buffer_type =
        (*(*input_buffer_requirements)->SupportedTypes())[0];
    auto input_buffer = TensorBuffer::CreateManaged(
        tensor_buffer_type, *tensor_type,
        (*input_buffer_requirements)->BufferSize().Value());
    if (!input_buffer.HasValue()) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                input_buffer.Error().Message());
    }
    input_buffers.push_back(std::move(*input_buffer));
  }
  return litert::Expected<std::vector<TensorBuffer>>(std::move(input_buffers));
}

litert::Expected<std::vector<TensorBuffer>> CompiledModel::CreateOutputBuffers(
    absl::string_view signature_key) {
  std::vector<TensorBuffer> output_buffers;
  auto output_names = GetOutputNames(signature_key);
  output_buffers.reserve(output_names.size());

  for (const auto& output_name : output_names) {
    auto output_buffer_requirements =
        GetOutputBufferRequirements(signature_key, output_name);
    if (!output_buffer_requirements.HasValue()) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                output_buffer_requirements.Error().Message());
    }
    auto tensor_type = GetOutputTensorType(signature_key, output_name);
    if (!tensor_type.HasValue()) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                tensor_type.Error().Message());
    }
    LiteRtTensorBufferType tensor_buffer_type =
        (*(*output_buffer_requirements)->SupportedTypes())[0];
    auto output_buffer = TensorBuffer::CreateManaged(
        tensor_buffer_type, *tensor_type,
        (*output_buffer_requirements)->BufferSize().Value());
    if (!output_buffer.HasValue()) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                output_buffer.Error().Message());
    }
    output_buffers.push_back(std::move(*output_buffer));
  }
  return litert::Expected<std::vector<TensorBuffer>>(std::move(output_buffers));
}

tflite::SignatureRunner* CompiledModel::GetSignatureRunner(
    absl::string_view signature_key) {
  if (signature_runners_.contains(signature_key)) {
    return signature_runners_[signature_key];
  }
  auto runner =
      interp_->GetSignatureRunner(signature_key == kDefaultSignatureKey
                                      ? nullptr
                                      : std::string(signature_key).c_str());
  signature_runners_[signature_key] = runner;
  return runner;
}

litert::Expected<void> CompiledModel::Invoke(
    absl::string_view signature_key, std::vector<TensorBuffer>& input_buffers,
    std::vector<TensorBuffer>& output_buffers) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Failed to get signature runner");
  }
  if (input_buffers.size() != runner->input_names().size()) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Input buffer size mismatch");
  }
  if (output_buffers.size() != runner->output_names().size()) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Output buffer size mismatch");
  }

  for (int i = 0; i < runner->input_names().size(); ++i) {
    const auto& input_name = runner->input_names()[i];
    auto* input_tensor = runner->input_tensor(input_name);
    if (input_buffers[i].BufferType().Value() ==
        kLiteRtTensorBufferTypeHostMemory) {
      // Assign CPU buffer via CustomAllocation.
      auto lock_and_addr =
          litert::TensorBufferScopedLock::Create(input_buffers[i]);
      TfLiteCustomAllocation custom_allocation{lock_and_addr->second,
                                               input_tensor->bytes};
      runner->SetCustomAllocationForInputTensor(input_name, custom_allocation,
                                                /*flags=*/0);
    } else {
      // Register tensor buffer for non CPU buffers.
      auto duplicate_buffer = input_buffers[i].Duplicate();

      if (auto status = buffer_context_->RegisterTensorBuffer(
              input_tensor, std::move(*duplicate_buffer));
          status != kLiteRtStatusOk) {
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "Failed to register input tensor buffer");
      }
    }
  }

  for (int i = 0; i < runner->output_names().size(); ++i) {
    const auto& output_name = runner->output_names()[i];
    auto* output_tensor = runner->output_tensor(output_name);
    if (output_buffers[i].BufferType().Value() ==
        kLiteRtTensorBufferTypeHostMemory) {
      // Assign CPU buffer via CustomAllocation.
      auto lock_and_addr =
          litert::TensorBufferScopedLock::Create(output_buffers[i]);
      TfLiteCustomAllocation custom_allocation{lock_and_addr->second,
                                               output_tensor->bytes};
      runner->SetCustomAllocationForOutputTensor(output_name, custom_allocation,
                                                 /*flags=*/0);
    } else {
      // Register tensor buffer for non CPU buffers.
      auto duplicate_buffer = output_buffers[i].Duplicate();

      if (auto status = buffer_context_->RegisterTensorBuffer(
              output_tensor, std::move(*duplicate_buffer));
          status != kLiteRtStatusOk) {
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "Failed to register output tensor buffer");
      }
    }
  }

  if (auto res = runner->AllocateTensors(); res != kTfLiteOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to allocate tensors");
  }

  if (auto res = runner->Invoke(); res != kTfLiteOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to invoke");
  }
  return litert::Expected<void>();
}

}  // namespace litert
