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

#include "xla/service/op_cost.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

// Used in LOG(INFO) statements for analysis logging.
constexpr std::string_view kLoggingAnalysisId = "COST_LOGGING";

}  // namespace

CostMetricId CostMetricId::LatencySeconds(const HloInstruction& instruction) {
  return CostMetricId(MetricType::kLatencySeconds, instruction, std::nullopt,
                      std::nullopt);
}

CostMetricId CostMetricId::ComputeSeconds(const HloInstruction& instruction) {
  return CostMetricId(MetricType::kComputeSeconds, instruction, std::nullopt,
                      std::nullopt);
}

CostMetricId CostMetricId::OperandBytesAccessed(
    const HloInstruction& instruction, int64_t operand_num,
    const ShapeIndex& shape_index) {
  return CostMetricId(MetricType::kOperandBytesAccessed, instruction,
                      operand_num, shape_index);
}

CostMetricId CostMetricId::OutputBytesAccessed(
    const HloInstruction& instruction, const ShapeIndex& shape_index) {
  return CostMetricId(MetricType::kOutputBytesAccessed, instruction,
                      std::nullopt, shape_index);
}

CostMetricId CostMetricId::TotalBytesAccessed(
    const HloInstruction& instruction) {
  return CostMetricId(MetricType::kTotalBytesAccessed, instruction,
                      std::nullopt, std::nullopt);
}

std::vector<std::string> CostMetricId::LoggingColumnNames() {
  return {"metric_id",        "metric_type", "module_name", "instruction_name",
          "instruction_type", "operand_num", "shape_index"};
}

bool CostMetricId::operator==(const CostMetricId& other) const {
  return MakeTuple() == other.MakeTuple();
}

int64_t CostMetricId::operand_num() const {
  CHECK(operand_num_.has_value());
  return *operand_num_;
}

const ShapeIndex& CostMetricId::shape_index() const {
  CHECK(shape_index_.has_value());
  return *shape_index_;
}

std::vector<std::string> CostMetricId::LoggingColumns() const {
  return {Identifier(),         metric_type_name(),
          ModuleName(),         std::string(instruction_->name()),
          InstructionTypeStr(), OperandNumStr(),
          ShapeIndexStr()};
}

std::string CostMetricId::ToString() const {
  return absl::StrCat(
      "<type=", metric_type_name(), ",computation=", ComputationName(),
      ",instruction=", instruction_->name(), ",operand_num=", OperandNumStr(),
      ",shape_index=", ShapeIndexStr(), ">");
}

CostMetricId::CostMetricId(MetricType type, const HloInstruction& instruction,
                           std::optional<int64_t> operand_num,
                           std::optional<ShapeIndex> shape_index)
    : type_(type),
      instruction_(&instruction),
      operand_num_(operand_num),
      shape_index_(std::move(shape_index)) {}

std::string CostMetricId::Identifier() const {
  std::string result;

  absl::Base64Escape(
      absl::StrJoin({absl::StrCat(static_cast<uint8_t>(type_)), ModuleName(),
                     absl::StrCat(instruction_->unique_id()), OperandNumStr(),
                     ShapeIndexStr()},
                    ","),
      &result);

  return result;
}

std::string CostMetricId::metric_type_name() const {
  switch (type_) {
    case MetricType::kLatencySeconds:
      return "latency-seconds";
    case MetricType::kComputeSeconds:
      return "compute-seconds";
    case MetricType::kOperandBytesAccessed:
      return "operand-bytes-accessed";
    case MetricType::kOutputBytesAccessed:
      return "output-bytes-accessed";
    case MetricType::kTotalBytesAccessed:
      return "total-bytes-accessed";
  }
}

std::string CostMetricId::ModuleName() const {
  if (instruction_->GetModule()) {
    return instruction_->GetModule()->name();
  }
  return "-";
}

std::string CostMetricId::ComputationName() const {
  if (instruction_->parent()) {
    return std::string(instruction_->parent()->name());
  }
  return "-";
}

std::string CostMetricId::InstructionTypeStr() const {
  if (instruction_->opcode() == HloOpcode::kCustomCall) {
    return absl::StrCat(HloOpcodeString(instruction_->opcode()), "-",
                        instruction_->custom_call_target());
  }

  if (instruction_->opcode() == HloOpcode::kFusion) {
    return absl::StrCat(HloOpcodeString(instruction_->opcode()), "-",
                        ::xla::ToString(instruction_->fusion_kind()));
  }

  return std::string(HloOpcodeString(instruction_->opcode()));
}

std::string CostMetricId::OperandNumStr() const {
  if (operand_num_.has_value()) {
    return absl::StrCat(*operand_num_);
  }
  return "-";
}

std::string CostMetricId::ShapeIndexStr() const {
  if (shape_index_.has_value()) {
    return shape_index_->ToString();
  }
  return "-";
}

CostMetricId::Tuple CostMetricId::MakeTuple() const {
  return std::make_tuple(type_, instruction_, operand_num_, shape_index_);
}

CostValue CostValue::MakeNotFound() { return CostValue(Type::kNotFound, 0.0); }

CostValue CostValue::MakeError() { return CostValue(Type::kError, 0.0); }

CostValue CostValue::MakeValue(double value) {
  return CostValue(Type::kOk, value);
}

bool CostValue::operator==(const CostValue& other) const {
  return MakeTuple() == other.MakeTuple();
}

double CostValue::value() const {
  CHECK(type_ == Type::kOk);
  return value_;
}

std::string CostValue::ToString() const {
  switch (type_) {
    case Type::kNotFound:
      return "nf";
    case Type::kError:
      return "err";
    case Type::kOk:
      return absl::StrCat(value_);
  }
}

namespace {

// Implementation for leaf calculation nodes.
class CalculationLeaf : public OpCostManager::CalculationNode {
 public:
  // If enable_cache is true, the leaf node will cache the MetricCalculators
  // it creates per HLO instruction.
  CalculationLeaf(absl::string_view name, OpCostCalculator calculator,
                  bool enable_cache, AcceptCostFn accept_cost_fn)
      : name_(name),
        calculator_(std::move(calculator)),
        enable_cache_(enable_cache) {
    if (accept_cost_fn) {
      accept_cost_fn_ = std::move(accept_cost_fn);
    } else {
      accept_cost_fn_ = [](const CostMetricId& metric_id, double cost) {
        return true;
      };
    }
  }

  ~CalculationLeaf() override = default;

  std::optional<double> GetMetricValue(
      const CostMetricId& metric_id,
      LeafCalculatorValueMap* calculator_value_map) override {
    MetricCalculator* calculator = nullptr;

    // Check the calculator cost cache.
    if (enable_cache_) {
      auto it = cached_costs_.find(&metric_id.instruction());
      if (it != cached_costs_.end()) {
        calculator = &it->second;
        VLOG(4) << "Found " << name_ << " calculator op cost in cache for "
                << metric_id.instruction().name();
      }
    }

    // If we didn't find an op cost in the cache, calculate it, and update the
    // cache (if enabled).
    MetricCalculator calculator_storage;
    if (!calculator) {
      calculator_storage = calculator_(metric_id.instruction());
      calculator = &calculator_storage;
      if (enable_cache_) {
        auto it_bool_pair = cached_costs_.insert(
            {&metric_id.instruction(), std::move(calculator_storage)});
        CHECK(it_bool_pair.second);
        calculator = &it_bool_pair.first->second;
        VLOG(4) << "Added " << name_ << " calculator op cost to cache for "
                << metric_id.instruction().name();
      }
    }

    // Get the CostValue.
    CostValue cost_value = (*calculator)(metric_id);
    if (calculator_value_map) {
      CHECK(calculator_value_map->insert({name_, cost_value}).second)
          << "Duplicate leaf calculator name " << name_;
    }

    if (!AcceptCalculatorCost(metric_id, cost_value)) {
      VLOG(2) << "Rejected leaf calculator value " << cost_value.ToString()
              << " for " << metric_id.ToString() << " from " << name_;
      return std::nullopt;
    }

    VLOG(1) << "Accepted leaf calculator value " << cost_value.ToString()
            << " for " << metric_id.ToString() << " from " << name_;
    return cost_value.value();
  }

  std::string_view Name() const override { return name_; }

  std::vector<std::string> LeafCalculatorNames() const override {
    return {name_};
  }

 private:
  bool AcceptCalculatorCost(const CostMetricId& metric_id,
                            const CostValue& cost_value) {
    if (!cost_value.IsOk()) {
      return false;
    }

    return accept_cost_fn_(metric_id, cost_value.value());
  }

  std::string name_;
  OpCostCalculator calculator_;
  bool enable_cache_;
  AcceptCostFn accept_cost_fn_;
  absl::flat_hash_map<const HloInstruction*, MetricCalculator> cached_costs_;
};

// Implementation for delegation calculation nodes.
class DelegationCalculationNode : public OpCostManager::CalculationNode {
 public:
  DelegationCalculationNode(
      absl::string_view name,
      std::vector<std::unique_ptr<OpCostManager::CalculationNode>> children,
      DelegationOrderFn delegation_order_fn)
      : name_(name), children_(std::move(children)) {
    if (delegation_order_fn) {
      delegation_order_fn_ = std::move(delegation_order_fn);
    } else {
      size_t num_children = children_.size();
      delegation_order_fn_ = [num_children](const HloInstruction& instruction,
                                            bool enable_analysis_logging) {
        DelegationInfo delegation_info;
        delegation_info.order.reserve(num_children);
        for (CalculatorIndex i = 0; i < num_children; ++i) {
          delegation_info.order.push_back(i);
        }
        return delegation_info;
      };
    }
  }

  ~DelegationCalculationNode() override = default;

  std::optional<double> GetMetricValue(
      const CostMetricId& metric_id,
      LeafCalculatorValueMap* calculator_value_map) override {
    bool analysis_logging_enabled = calculator_value_map;
    DelegationInfo delegation_info =
        delegation_order_fn_(metric_id.instruction(), analysis_logging_enabled);
    std::optional<double> final_result = std::nullopt;
    for (CalculatorIndex calculator_index : delegation_info.order) {
      CHECK_LT(calculator_index, children_.size());
      VLOG(3) << "Delegating to " << children_[calculator_index]->Name()
              << " to compute " << metric_id.ToString();
      std::optional<double> result =
          children_[calculator_index]->GetMetricValue(metric_id,
                                                      calculator_value_map);
      if (!final_result.has_value() && result.has_value()) {
        final_result = result.value();
        if (!analysis_logging_enabled) {
          break;
        }
      }
    }

    // Go through the remaining calculators for logging purposes.
    if (analysis_logging_enabled) {
      for (CalculatorIndex calculator_index :
           delegation_info.additional_calculators_to_log) {
        CHECK_LT(calculator_index, children_.size());
        children_[calculator_index]->GetMetricValue(metric_id,
                                                    calculator_value_map);
      }
    }

    return final_result;
  }

  std::string_view Name() const override { return name_; }

  std::vector<std::string> LeafCalculatorNames() const override {
    std::vector<std::string> result;
    for (const auto& child : children_) {
      std::vector<std::string> child_names = child->LeafCalculatorNames();
      result.insert(result.end(), child_names.begin(), child_names.end());
    }
    return result;
  }

 private:
  DelegationCalculationNode() = delete;

  std::string name_;
  std::vector<std::unique_ptr<OpCostManager::CalculationNode>> children_;
  DelegationOrderFn delegation_order_fn_;
};

}  // namespace

std::unique_ptr<OpCostManager::CalculationNode>
OpCostManager::CalculationNode::CreateLeaf(absl::string_view name,
                                           OpCostCalculator calculator,
                                           bool enable_cache,
                                           AcceptCostFn accept_cost_fn) {
  return std::make_unique<CalculationLeaf>(
      name, std::move(calculator), enable_cache, std::move(accept_cost_fn));
}

std::unique_ptr<OpCostManager::CalculationNode>
OpCostManager::CalculationNode::CreateDelegationNode(
    absl::string_view name,
    std::vector<std::unique_ptr<OpCostManager::CalculationNode>> children,
    DelegationOrderFn delegation_order_fn) {
  return std::make_unique<DelegationCalculationNode>(
      name, std::move(children), std::move(delegation_order_fn));
}

OpCostManager::OpCostManager(Options options,
                             std::unique_ptr<CalculationNode> root)
    : options_(std::move(options)),
      root_(std::move(root)),
      leaf_calculator_names_([&]() {
        std::vector<std::string> calculator_names =
            root_->LeafCalculatorNames();
        absl::c_sort(calculator_names);
        absl::string_view previous = "";
        for (const std::string& calculator_name : calculator_names) {
          CHECK_NE(calculator_name, previous);
          previous = calculator_name;
        }
        return calculator_names;
      }()) {
  LOG_IF(INFO, options_.enable_analysis_logging) << AnalysisLoggingColumns();
}

double OpCostManager::LatencySeconds(const HloInstruction& instruction) {
  return GetMetricValue(CostMetricId::LatencySeconds(instruction));
}

double OpCostManager::ComputeSeconds(const HloInstruction& instruction) {
  return GetMetricValue(CostMetricId::ComputeSeconds(instruction));
}

double OpCostManager::OperandBytesAccessed(const HloInstruction& instruction,
                                           int64_t operand_num,
                                           const ShapeIndex& shape_index) {
  return GetMetricValue(CostMetricId::OperandBytesAccessed(
      instruction, operand_num, shape_index));
}

double OpCostManager::OutputBytesAccessed(const HloInstruction& instruction,
                                          const ShapeIndex& shape_index) {
  return GetMetricValue(
      CostMetricId::OutputBytesAccessed(instruction, shape_index));
}

double OpCostManager::TotalBytesAccessed(const HloInstruction& instruction) {
  return GetMetricValue(CostMetricId::TotalBytesAccessed(instruction));
}

double OpCostManager::GetMetricValue(const CostMetricId& metric_id) {
  // Check the metric cache.
  if (options_.enable_cache) {
    auto it = metric_cache_.find(metric_id);
    if (it != metric_cache_.end()) {
      VLOG(4) << "Found cost for " << metric_id.ToString() << " in cache";
      return it->second;
    }
  }

  OpCostManager::CalculationNode::LeafCalculatorValueMap
      calculator_value_map_storage;
  OpCostManager::CalculationNode::LeafCalculatorValueMap* calculator_value_map =
      options_.enable_analysis_logging ? &calculator_value_map_storage
                                       : nullptr;

  VLOG(3) << "Delegating to " << root_->Name() << " to compute "
          << metric_id.ToString();
  std::optional<double> value =
      root_->GetMetricValue(metric_id, calculator_value_map);
  // If users don't want to crash, they should register a calculator that
  // computes a default cost.
  LOG_IF(FATAL, !value.has_value())
      << "Unable to compute a cost for " << metric_id.ToString();
  if (options_.enable_cache) {
    metric_cache_[metric_id] = value.value();
    VLOG(4) << "Added cost for " << metric_id.ToString() << " to the cache";
  }

  LOG_IF(INFO, options_.enable_analysis_logging)
      << AnalysisLoggingLine(metric_id, *calculator_value_map);

  return value.value();
}

std::string OpCostManager::AnalysisLoggingColumns() const {
  std::vector<std::string> columns = CostMetricId::LoggingColumnNames();
  columns.insert(columns.end(), leaf_calculator_names_.begin(),
                 leaf_calculator_names_.end());

  return absl::StrCat(kLoggingAnalysisId, ": ", absl::StrJoin(columns, "\t"));
}

std::string OpCostManager::AnalysisLoggingLine(
    const CostMetricId& metric_id,
    const OpCostManager::CalculationNode::LeafCalculatorValueMap&
        calculator_costs) const {
  std::vector<std::string> columns = metric_id.LoggingColumns();
  for (const std::string& calculator_name : leaf_calculator_names_) {
    auto it = calculator_costs.find(calculator_name);
    if (it != calculator_costs.end()) {
      columns.push_back(it->second.ToString());
    } else {
      columns.push_back("na");
    }
  }
  return absl::StrCat(kLoggingAnalysisId, ": ", absl::StrJoin(columns, "\t"));
}

HloCostAnalysisWithAcceptState::HloCostAnalysisWithAcceptState(
    std::unique_ptr<HloCostAnalysis> cost_analysis,
    bool accepted_entry_computation)
    : cost_analysis_storage_(std::move(cost_analysis)),
      cost_analysis_(*cost_analysis_storage_),
      accepted_entry_computation_(accepted_entry_computation) {}

HloCostAnalysisWithAcceptState::HloCostAnalysisWithAcceptState(
    HloCostAnalysis& cost_analysis, bool accepted_entry_computation)
    : cost_analysis_(cost_analysis),
      accepted_entry_computation_(accepted_entry_computation) {}

HloCostAnalysis& HloCostAnalysisWithAcceptState::cost_analysis(
    const HloInstruction& instruction) {
  if (!accepted_entry_computation_) {
    CHECK(instruction.GetModule());
    absl::Status status =
        instruction.GetModule()->entry_computation()->Accept(&cost_analysis_);
    LOG_IF(FATAL, !status.ok())
        << "Computation "
        << instruction.GetModule()->entry_computation()->name()
        << " failed to accept HloCostAnalysis: " << status;
    accepted_entry_computation_ = true;
  }

  return cost_analysis_;
}

namespace {

CostValue HloCostAnalysisLatencySeconds(
    const HloCostAnalysis& hlo_cost_analysis,
    const HloInstruction& instruction) {
  std::vector<double> latencies = {
      // Min latency;
      hlo_cost_analysis.min_latency_seconds(HloCostAnalysis::kFlopsKey),
      // Latency.
      hlo_cost_analysis.optimal_seconds(instruction)};
  return CostValue::MakeValue(*absl::c_max_element(latencies));
}

CostValue HloCostAnalysisComputeSeconds(
    const HloCostAnalysis& hlo_cost_analysis,
    const HloInstruction& instruction) {
  std::vector<double> latencies = {
      // Min latency;
      hlo_cost_analysis.min_latency_seconds(HloCostAnalysis::kFlopsKey),
      // Standard compute latency.
      static_cast<double>(hlo_cost_analysis.flop_count(instruction)) /
          static_cast<double>(
              hlo_cost_analysis.per_second_rate(HloCostAnalysis::kFlopsKey)),
      // Transcendental compute latency.
      static_cast<double>(hlo_cost_analysis.transcendental_count(instruction)) /
          static_cast<double>(hlo_cost_analysis.per_second_rate(
              HloCostAnalysis::kTranscendentalsKey))};
  return CostValue::MakeValue(*absl::c_max_element(latencies));
}

CostValue HloCostAnalysisOperandBytesAccessed(
    const HloCostAnalysis& hlo_cost_analysis, const HloInstruction& instruction,
    int64_t operand_num, const ShapeIndex& shape_index) {
  return CostValue::MakeValue(
      static_cast<double>(hlo_cost_analysis.operand_bytes_accessed(
          instruction, operand_num, shape_index)));
}

CostValue HloCostAnalysisOutputBytesAccessed(
    const HloCostAnalysis& hlo_cost_analysis, const HloInstruction& instruction,
    const ShapeIndex& shape_index) {
  return CostValue::MakeValue(static_cast<double>(
      hlo_cost_analysis.output_bytes_accessed(instruction, shape_index)));
}

CostValue HloCostAnalysisTotalBytesAccessed(
    const HloCostAnalysis& hlo_cost_analysis,
    const HloInstruction& instruction) {
  return CostValue::MakeValue(
      static_cast<double>(hlo_cost_analysis.bytes_accessed(instruction)));
}

}  // namespace

OpCostCalculator CreateHloCostAnalysisCalculator(
    HloCostAnalysisWithAcceptState& cost_analysis_wrapper) {
  OpCostCalculator result =
      [&cost_analysis_wrapper](const HloInstruction& instruction) {
        const HloCostAnalysis& hlo_cost_analysis =
            cost_analysis_wrapper.cost_analysis(instruction);

        MetricCalculator metric_cost_calculator =
            [&hlo_cost_analysis](const CostMetricId& metric_id) {
              switch (metric_id.type()) {
                case CostMetricId::MetricType::kLatencySeconds:
                  return HloCostAnalysisLatencySeconds(hlo_cost_analysis,
                                                       metric_id.instruction());
                case CostMetricId::MetricType::kComputeSeconds:
                  return HloCostAnalysisComputeSeconds(hlo_cost_analysis,
                                                       metric_id.instruction());
                case CostMetricId::MetricType::kOperandBytesAccessed:
                  return HloCostAnalysisOperandBytesAccessed(
                      hlo_cost_analysis, metric_id.instruction(),
                      metric_id.operand_num(), metric_id.shape_index());
                case CostMetricId::MetricType::kOutputBytesAccessed:
                  return HloCostAnalysisOutputBytesAccessed(
                      hlo_cost_analysis, metric_id.instruction(),
                      metric_id.shape_index());
                case CostMetricId::MetricType::kTotalBytesAccessed:
                  return HloCostAnalysisTotalBytesAccessed(
                      hlo_cost_analysis, metric_id.instruction());
              };
            };

        return metric_cost_calculator;
      };

  return result;
}

namespace {

CostValue DefaultTotalBytesAccessedComputation(
    const HloInstruction& instruction, MetricCalculator& metric_calculator) {
  CostValue result = CostValue::MakeValue(0.0);
  auto update_result = [&result](const Shape& subshape, CostValue next_cost) {
    if (!result.IsOk()) {
      return;
    }
    if (next_cost.IsNotFound()) {
      result = CostValue::MakeNotFound();
      return;
    }
    if (next_cost.IsError()) {
      result = CostValue::MakeError();
      return;
    }
    result = CostValue::MakeValue(result.value() + next_cost.value());
  };

  for (int64_t operand_num = 0; operand_num < instruction.operand_count();
       ++operand_num) {
    const HloInstruction& operand = *instruction.operand(operand_num);
    ShapeUtil::ForEachSubshape(
        operand.shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsTuple()) {
            return;
          }
          update_result(subshape,
                        metric_calculator(CostMetricId::OperandBytesAccessed(
                            instruction, operand_num, index)));
        });
  }
  ShapeUtil::ForEachSubshape(instruction.shape(), [&](const Shape& subshape,
                                                      const ShapeIndex& index) {
    if (subshape.IsTuple()) {
      return;
    }
    update_result(subshape, metric_calculator(CostMetricId::OutputBytesAccessed(
                                instruction, index)));
  });

  return result;
}

// Returns an OpCostCalculator that generates MetricCalculators using
// initial_calculator, and then post-processes them using post_processor.
OpCostCalculator CreateCalculatorWithPostProcessedMetricCalculator(
    OpCostCalculator initial_calculator,
    absl::AnyInvocable<MetricCalculator(MetricCalculator)> post_processor) {
  OpCostCalculator result = [initial_calculator = std::move(initial_calculator),
                             post_processor = std::move(post_processor)](
                                const HloInstruction& instruction) mutable {
    MetricCalculator initial_metric_calculator =
        initial_calculator(instruction);

    return post_processor(std::move(initial_metric_calculator));
  };

  return result;
}

}  // namespace

OpCostCalculator CreateCalculatorWithDefaultTotalBytesAccessed(
    OpCostCalculator initial_calculator) {
  return CreateCalculatorWithPostProcessedMetricCalculator(
      std::move(initial_calculator), [](MetricCalculator metric_calculator) {
        return [metric_calculator = std::move(metric_calculator)](
                   const CostMetricId& metric_id) mutable {
          if (metric_id.type() ==
              CostMetricId::MetricType::kTotalBytesAccessed) {
            return DefaultTotalBytesAccessedComputation(metric_id.instruction(),
                                                        metric_calculator);
          }
          return metric_calculator(metric_id);
        };
      });
}

OpCostCalculator CreateCalculatorWithPostProcessedCostValues(
    OpCostCalculator initial_calculator,
    absl::AnyInvocable<CostValue(const CostMetricId& metric_id,
                                 CostValue cost_value)>
        post_process_cost_value) {
  return CreateCalculatorWithPostProcessedMetricCalculator(
      std::move(initial_calculator),
      // This lambda takes a MetricCalculator and returns a new MetricCalculator
      // that post-processes the CostValues.
      [post_process_cost_value = std::move(post_process_cost_value)](
          MetricCalculator metric_calculator) mutable {
        return
            // We capture post_process_cost_value by reference, so that it can
            // be used in every MetricCalculator we generate here.
            [metric_calculator = std::move(metric_calculator),
             &post_process_cost_value](const CostMetricId& metric_id) mutable {
              return post_process_cost_value(metric_id,
                                             metric_calculator(metric_id));
            };
      });
}

}  // namespace xla
