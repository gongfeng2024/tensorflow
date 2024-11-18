/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/python/dlpack_strides.h"

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(DlpackStridesTest, basic) {
  std::vector<int64_t> dims = {2, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};
  auto layout = StridesToLayout(absl::MakeSpan(dims), absl::MakeSpan(strides));
  EXPECT_TRUE(layout.ok());
  EXPECT_EQ(layout.value(), std::vector<int64_t>({2, 1, 0}));

  std::vector<int64_t> strides_cm = {1, 2, 6};
  auto layout_cm =
      StridesToLayout(absl::MakeSpan(dims), absl::MakeSpan(strides_cm));
  EXPECT_TRUE(layout_cm.ok());
  EXPECT_EQ(layout_cm.value(), std::vector<int64_t>({0, 1, 2}));
}

TEST(DlpackStridesTest, unitDim) {
  // Row-major
  std::vector<int64_t> dims = {2, 1, 3, 4};
  std::vector<int64_t> strides = {12, 12, 4, 1};
  auto layout = StridesToLayout(absl::MakeSpan(dims), absl::MakeSpan(strides));
  EXPECT_TRUE(layout.ok());
  EXPECT_EQ(layout.value(), std::vector<int64_t>({3, 2, 1, 0}));

  std::vector<int64_t> strides2 = {12, 1, 4, 1};
  auto layout2 =
      StridesToLayout(absl::MakeSpan(dims), absl::MakeSpan(strides2));
  EXPECT_TRUE(layout2.ok());
  EXPECT_EQ(layout2.value(), std::vector<int64_t>({3, 2, 1, 0}));

  // Column-major. Note that in these cases, since one of the dimensions is 1,
  // there are several valid layouts that we could produce. We choose to prefer
  // row-major whenever there are multiple valid layouts, so the output layouts
  // here aren't completely column-major.
  std::vector<int64_t> strides_cm = {1, 2, 2, 6};
  auto layout_cm =
      StridesToLayout(absl::MakeSpan(dims), absl::MakeSpan(strides_cm));
  EXPECT_TRUE(layout_cm.ok());
  EXPECT_EQ(layout_cm.value(), std::vector<int64_t>({1, 0, 2, 3}));

  std::vector<int64_t> strides2_cm = {1, 1, 2, 6};
  auto layout2_cm =
      StridesToLayout(absl::MakeSpan(dims), absl::MakeSpan(strides2_cm));
  EXPECT_TRUE(layout2_cm.ok());
  EXPECT_EQ(layout2_cm.value(), std::vector<int64_t>({1, 0, 2, 3}));
}

}  // namespace
}  // namespace xla
