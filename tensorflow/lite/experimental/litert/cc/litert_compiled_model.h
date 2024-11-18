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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_COMPILED_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_COMPILED_MODEL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/runtime/external_litert_buffer_context.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

namespace litert {

// The CompiledModel is a higher level inference API.  It is created by
// provided model with compilation options. Internally, it instanciates runtime
// and applies Delegates mapped to the complication options.
// It also supports getting BufferRequirements to create input/output
// TensorBuffers, and it allows to invoke the model with the input/output
// TensorBuffers.
//
// Example user flow:
//
// 1. Create CompiledModel
// 2. Query the model input/output requirements
// 3. Create input/output TensorBuffers
// 4. Fill the input TensorBuffers with input data
// 5. Invoke the model with the input/output TensorBuffers
// 6. Evaluate the output TensorBuffers
//
// TODO: b/379317134 - Support compilation options once LiteRtAccelerator is
// ready.
class CompiledModel {
 public:
  using Ptr = std::unique_ptr<CompiledModel>;

  // Creates a CompiledModel from a TFLite file.
  // The created CompiledModel only runs with Xnnpack delegate.
  // The model is loaded into memory and the caller takes ownership of the
  // returned object.
  // WARNING: This API will be deprecated once LiteRtAccelerator is ready.
  static Expected<Ptr> CreateFromTflFile(absl::string_view filename);

  // Similar to CreateFromTflFile, but the model runs with DispatchDelegate.
  // WARNING: This API will be deprecated once LiteRtAccelerator is ready.
  static Expected<Ptr> CreateFromTflFileWithByteCode(
      absl::string_view tfl_filename, absl::string_view npu_filename);

  // Returns the list of signatures defined in the model.
  const std::vector<absl::string_view>& GetSignatures() const {
    return signature_keys_;
  }

  // Returns the list of input tensor names for the given signature.
  std::vector<const char*> GetInputNames(absl::string_view signature_key);

  // Returns the list of output tensor names for the given signature.
  std::vector<const char*> GetOutputNames(absl::string_view signature_key);

  // Returns the buffer requirements for the given input tensor. The returned
  // TensorBufferRequirements object is used to create the input tensor buffer.
  litert::Expected<TensorBufferRequirements*> GetInputBufferRequirements(
      absl::string_view signature_key, absl::string_view input_name);

  // Returns the buffer requirements for the given output tensor. The returned
  // TensorBufferRequirements object is used to create the output tensor buffer.
  litert::Expected<TensorBufferRequirements*> GetOutputBufferRequirements(
      absl::string_view signature_key, absl::string_view output_name);

  // Returns the RankedTensorType for the given input tensor name.
  // This is used to create the input tensor buffer.
  litert::Expected<RankedTensorType> GetInputTensorType(
      absl::string_view signature_key, absl::string_view input_name);

  // Returns the RankedTensorType for the given output tensor name.
  // This is used to create the output tensor buffer.
  litert::Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view signature_key, absl::string_view output_name);

  // A helper function to creates the input tensor buffers for the given
  // signature. It uses BufferRequirements and RankedTensorType to create the
  // input tensor buffers.
  litert::Expected<std::vector<TensorBuffer>> CreateInputBuffers(
      absl::string_view signature_key);

  // A helper function to creates the output tensor buffers for the given
  // signature. It uses BufferRequirements and RankedTensorType to create the
  // output tensor buffers.
  litert::Expected<std::vector<TensorBuffer>> CreateOutputBuffers(
      absl::string_view signature_key);

  // Invokes the model of the given signature with the provided input/output
  // TensorBuffers.
  litert::Expected<void> Invoke(absl::string_view signature_key,
                                std::vector<TensorBuffer>& input_buffers,
                                std::vector<TensorBuffer>& output_buffers);

  // Default signature key. This is the key that is used if the model does not
  // define any signatures.
  static constexpr absl::string_view kDefaultSignatureKey = "main";

 private:
  // Processes the model and initializes the internal states.
  // This is called in the public Create*() methods.
  Expected<void> Initialize();

  // Returns the SignatureRunner for the given signature key.
  // If the signature key is not found, returns nullptr.
  tflite::SignatureRunner* GetSignatureRunner(absl::string_view signature_key);

  std::unique_ptr<::tflite::Interpreter> interp_;
  std::unique_ptr<::tflite::FlatBufferModel> fb_model_;
  std::unique_ptr<::tflite::Allocation> alloc_;
  OwningBufferRef<uint8_t> model_buf_;

  // The ExternalLiteRtBufferContext used to register tensor buffers with
  // Delegates.
  std::unique_ptr<litert::internal::ExternalLiteRtBufferContext>
      buffer_context_;

  // The list of signature keys defined in the model. This is the same as
  // SignatureRunner::signature_keys() but storring absl::string_view instead of
  // pointer.
  std::vector<absl::string_view> signature_keys_;

  // Map from signature key to SignatureRunner. This is used to lazy calling
  // GetSignatureRunner() which is expensive.
  absl::flat_hash_map<absl::string_view, tflite::SignatureRunner*>
      signature_runners_;

  // The buffer requirement maps for CPU buffers. For delegates with CPU
  // buffers, they don't register TensorBufferRequirements. Instead, the
  // CompiledModel creates the TensorBufferRequirements and stores them in this
  // map.
  absl::flat_hash_map<const TfLiteTensor*, TensorBufferRequirements>
      cpu_buffer_requirements_;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_COMPILED_MODEL_H_
