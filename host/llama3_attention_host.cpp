#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>
#include <version.h>
#include <experimental/xrt_system.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace fs = std::filesystem;

namespace {

#ifndef LLAMA3_HOST_VERSION
#define LLAMA3_HOST_VERSION "1.0.0"
#endif
#ifndef LLAMA3_DESIGN_VERSION
#define LLAMA3_DESIGN_VERSION "Llama3-AIE-FlashStream"
#endif
#ifndef LLAMA3_KERNEL_NAME
#define LLAMA3_KERNEL_NAME "llama3_attention"
#endif
#ifndef LLAMA3_QK_COMPUTE_LABEL
#define LLAMA3_QK_COMPUTE_LABEL "INT8 x INT8 -> INT32"
#endif
#ifndef LLAMA3_AIE_TILES
#define LLAMA3_AIE_TILES 64
#endif
#ifndef LLAMA3_AIE_TILE_BUDGET
#define LLAMA3_AIE_TILE_BUDGET 64
#endif
#ifndef LLAMA3_OUTPUT_PLANES
#define LLAMA3_OUTPUT_PLANES 4
#endif

constexpr const char* kHostVersion = LLAMA3_HOST_VERSION;
constexpr const char* kDesignVersion = LLAMA3_DESIGN_VERSION;
constexpr const char* kKernelName = LLAMA3_KERNEL_NAME;
constexpr int kSequenceLength = 32;
constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kAieTiles = LLAMA3_AIE_TILES;
constexpr int kAieTileBudget = LLAMA3_AIE_TILE_BUDGET;
constexpr int kOutputPlanes = LLAMA3_OUTPUT_PLANES;
constexpr int kPlFrequencyMhz = 300;
constexpr double kFunctionalErrorLimit = 0.05;
constexpr double kInternalErrorTarget = 0.003;
static_assert(kOutputPlanes == 1 || kOutputPlanes == 4,
              "Host supports either one contiguous output or four 32-D planes");

class HardwareError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Options {
  std::string xclbin;
  int batch = 1;
  int warmup = 5;
  int runs = 25;
  std::uint32_t seed = 7;
  bool verify = true;
  bool profile = false;
  bool quiet = false;
  std::string output_json;
  std::string output_csv;
};

struct Statistics {
  double minimum = 0.0;
  double mean = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double maximum = 0.0;
  double standard_deviation = 0.0;
};

struct TimingSamples {
  std::vector<double> h2d;
  std::vector<double> kernel;
  std::vector<double> d2h;
  std::vector<double> e2e;
};

struct Accuracy {
  bool measured = false;
  bool functional_pass = false;
  bool internal_003_pass = false;
  double mean_abs_error = 0.0;
  double p99_abs_error = 0.0;
  double p999_abs_error = 0.0;
  double max_abs_error = 0.0;
  double max_abs_output = 0.0;
  int worst_batch = 0;
  int worst_head = 0;
  int worst_row = 0;
  int worst_dimension = 0;
  double reference_value = 0.0;
  double hardware_value = 0.0;
};

struct Metadata {
  std::string xclbin_path;
  std::string xclbin_sha256;
  std::string build_time = std::string(__DATE__) + " " + __TIME__;
  std::string xrt_version = xrt_build_version;
  std::string device_bdf = "N/A";
  std::string device_name = "N/A";
};

struct BenchmarkResult {
  TimingSamples samples;
  Statistics h2d;
  Statistics kernel;
  Statistics d2h;
  Statistics e2e;
  Accuracy accuracy;
  Metadata metadata;
  double attention_flops = 0.0;
  double kernel_gflops = 0.0;
  double e2e_gflops = 0.0;
  double requests_per_second = 0.0;
  double tokens_per_second = 0.0;
};

std::string usage(const char* program) {
  std::ostringstream out;
  out << "Usage: " << program << " --xclbin PATH [options]\n\n"
      << "Required:\n"
      << "  --xclbin PATH       Accelerator xclbin path\n\n"
      << "Options:\n"
      << "  --batch N           Requests per kernel launch (default: 1)\n"
      << "  --warmup N          Warmup launches (default: 5)\n"
      << "  --runs N            Measured launches (default: 25)\n"
      << "  --seed N            mt19937 input seed (default: 7)\n"
      << "  --verify             Enable full CPU reference (default)\n"
      << "  --no-verify          Skip CPU reference and accuracy metrics\n"
      << "  --profile            Print one compact line per measured run\n"
      << "  --quiet              Print only the final result table\n"
      << "  --output-json PATH   Write aggregate machine-readable results\n"
      << "  --output-csv PATH    Write raw per-run latency samples\n"
      << "  --version            Print Host and design versions\n"
      << "  --help               Show this help\n";
  return out.str();
}

int parse_positive(const std::string& text, const char* option) {
  std::size_t used = 0;
  long value = 0;
  try {
    value = std::stol(text, &used, 10);
  } catch (...) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  if (used != text.size() || value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(option) + " must be a positive integer");
  }
  return static_cast<int>(value);
}

int parse_nonnegative(const std::string& text, const char* option) {
  std::size_t used = 0;
  long value = 0;
  try {
    value = std::stol(text, &used, 10);
  } catch (...) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  if (used != text.size() || value < 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(option) + " must be a non-negative integer");
  }
  return static_cast<int>(value);
}

std::uint32_t parse_seed(const std::string& text) {
  std::size_t used = 0;
  unsigned long value = 0;
  try {
    value = std::stoul(text, &used, 10);
  } catch (...) {
    throw std::invalid_argument("--seed requires an unsigned 32-bit integer");
  }
  if (used != text.size() || value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("--seed requires an unsigned 32-bit integer");
  }
  return static_cast<std::uint32_t>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  auto value_after = [&](int& index, const char* option) -> std::string {
    if (++index >= argc) throw std::invalid_argument(std::string(option) + " requires a value");
    return argv[index];
  };

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      std::cout << usage(argv[0]);
      std::exit(0);
    } else if (argument == "--version") {
      std::cout << "AIE-FlashStream Host " << kHostVersion
                << " (design " << kDesignVersion << ")\n";
      std::exit(0);
    } else if (argument == "--xclbin") {
      options.xclbin = value_after(index, "--xclbin");
    } else if (argument == "--batch") {
      options.batch = parse_positive(value_after(index, "--batch"), "--batch");
    } else if (argument == "--warmup") {
      options.warmup = parse_nonnegative(value_after(index, "--warmup"), "--warmup");
    } else if (argument == "--runs") {
      options.runs = parse_positive(value_after(index, "--runs"), "--runs");
    } else if (argument == "--seed") {
      options.seed = parse_seed(value_after(index, "--seed"));
    } else if (argument == "--verify") {
      options.verify = true;
    } else if (argument == "--no-verify") {
      options.verify = false;
    } else if (argument == "--profile") {
      options.profile = true;
    } else if (argument == "--quiet") {
      options.quiet = true;
    } else if (argument == "--output-json") {
      options.output_json = value_after(index, "--output-json");
    } else if (argument == "--output-csv") {
      options.output_csv = value_after(index, "--output-csv");
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }

  if (options.xclbin.empty()) throw std::invalid_argument("--xclbin is required");
  const fs::path xclbin_path(options.xclbin);
  if (!fs::exists(xclbin_path)) throw std::invalid_argument("xclbin does not exist: " + options.xclbin);
  if (!fs::is_regular_file(xclbin_path)) throw std::invalid_argument("xclbin is not a regular file: " + options.xclbin);
  return options;
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>(bits >> 16);
}

float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::size_t q_elements(int batch) {
  return static_cast<std::size_t>(batch) * kQueryHeads * kSequenceLength * kHeadDim;
}

std::size_t kv_elements(int batch) {
  return static_cast<std::size_t>(batch) * kKvHeads * kSequenceLength * kHeadDim;
}

std::size_t q_offset(int batch, int head, int row, int dimension) {
  return (((static_cast<std::size_t>(batch) * kQueryHeads + head) *
           kSequenceLength + row) * kHeadDim + dimension);
}

std::size_t kv_offset(int batch, int head, int row, int dimension) {
  return (((static_cast<std::size_t>(batch) * kKvHeads + head) *
           kSequenceLength + row) * kHeadDim + dimension);
}

void fill_random(std::uint16_t* data, std::size_t count, std::mt19937& rng) {
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (std::size_t index = 0; index < count; ++index)
    data[index] = float_to_bf16(distribution(rng));
}

std::vector<float> reference_attention(const std::uint16_t* q,
                                       const std::uint16_t* k,
                                       const std::uint16_t* v, int batch_size) {
  std::vector<float> result(q_elements(batch_size));
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int q_head = 0; q_head < kQueryHeads; ++q_head) {
      const int kv_head = q_head / (kQueryHeads / kKvHeads);
      for (int query = 0; query < kSequenceLength; ++query) {
        float max_score = -std::numeric_limits<float>::infinity();
        for (int key = 0; key <= query; ++key) {
          float dot = 0.0f;
          for (int dim = 0; dim < kHeadDim; ++dim) {
            dot += bf16_to_float(q[q_offset(batch, q_head, query, dim)]) *
                   bf16_to_float(k[kv_offset(batch, kv_head, key, dim)]);
          }
          max_score = std::max(max_score, dot * scale);
        }
        float denominator = 0.0f;
        std::vector<float> weights(query + 1);
        for (int key = 0; key <= query; ++key) {
          float dot = 0.0f;
          for (int dim = 0; dim < kHeadDim; ++dim) {
            dot += bf16_to_float(q[q_offset(batch, q_head, query, dim)]) *
                   bf16_to_float(k[kv_offset(batch, kv_head, key, dim)]);
          }
          weights[key] = std::exp(dot * scale - max_score);
          denominator += weights[key];
        }
        for (int dim = 0; dim < kHeadDim; ++dim) {
          float numerator = 0.0f;
          for (int key = 0; key <= query; ++key) {
            numerator += weights[key] *
                         bf16_to_float(v[kv_offset(batch, kv_head, key, dim)]);
          }
          result[q_offset(batch, q_head, query, dim)] = numerator / denominator;
        }
      }
    }
  }
  return result;
}

double nearest_rank(std::vector<double> values, double quantile) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  if (quantile <= 0.0) return values.front();
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(quantile * static_cast<double>(values.size()))) - 1;
  return values[std::min(index, values.size() - 1)];
}

Statistics summarize(const std::vector<double>& values) {
  if (values.empty()) return {};
  Statistics result;
  result.minimum = *std::min_element(values.begin(), values.end());
  result.maximum = *std::max_element(values.begin(), values.end());
  result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  result.p50 = nearest_rank(values, 0.50);
  result.p90 = nearest_rank(values, 0.90);
  result.p95 = nearest_rank(values, 0.95);
  result.p99 = nearest_rank(values, 0.99);
  double squared_sum = 0.0;
  for (double value : values) squared_sum += (value - result.mean) * (value - result.mean);
  result.standard_deviation = std::sqrt(squared_sum / values.size());
  return result;
}

std::string shell_quote(const std::string& value) {
  std::string result = "'";
  for (char character : value) {
    if (character == '\'') result += "'\\''";
    else result += character;
  }
  return result + "'";
}

std::string sha256_file(const std::string& path) {
  const std::string command = "sha256sum -- " + shell_quote(path) + " 2>/dev/null";
  std::array<char, 256> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) return "N/A";
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) output += buffer.data();
  const int status = pclose(pipe);
  if (status != 0 || output.size() < 64) return "N/A";
  const std::string digest = output.substr(0, 64);
  if (!std::all_of(digest.begin(), digest.end(), [](unsigned char value) {
        return std::isxdigit(value) != 0;
      })) return "N/A";
  return digest;
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char character : value) {
    switch (character) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (character < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(character) << std::dec << std::setfill(' ');
        } else {
          out << character;
        }
    }
  }
  return out.str();
}

void ensure_parent(const std::string& path) {
  const fs::path parent = fs::path(path).parent_path();
  if (!parent.empty()) fs::create_directories(parent);
}

Accuracy evaluate_accuracy(const std::uint16_t* output,
                            const std::vector<float>& expected, int batch_size) {
  Accuracy result;
  result.measured = !expected.empty();
  std::vector<double> errors;
  if (result.measured) errors.reserve(expected.size());
  double error_sum = 0.0;
  std::size_t worst_index = 0;
  for (std::size_t index = 0; index < q_elements(batch_size); ++index) {
    const double actual = bf16_to_float(output[index]);
    result.max_abs_output = std::max(result.max_abs_output, std::abs(actual));
    if (!result.measured) continue;
    const double error = std::abs(actual - expected[index]);
    errors.push_back(error);
    error_sum += error;
    if (error > result.max_abs_error) {
      result.max_abs_error = error;
      worst_index = index;
      result.hardware_value = actual;
      result.reference_value = expected[index];
    }
  }
  if (!result.measured) return result;
  result.mean_abs_error = error_sum / errors.size();
  result.p99_abs_error = nearest_rank(errors, 0.99);
  result.p999_abs_error = nearest_rank(errors, 0.999);
  std::size_t coordinate = worst_index;
  result.worst_dimension = coordinate % kHeadDim;
  coordinate /= kHeadDim;
  result.worst_row = coordinate % kSequenceLength;
  coordinate /= kSequenceLength;
  result.worst_head = coordinate % kQueryHeads;
  result.worst_batch = coordinate / kQueryHeads;
  result.functional_pass = result.max_abs_error < kFunctionalErrorLimit;
  result.internal_003_pass = result.max_abs_error <= kInternalErrorTarget;
  return result;
}

BenchmarkResult run_benchmark(const Options& options) {
  BenchmarkResult result;
  result.metadata.xclbin_path = fs::path(options.xclbin).filename().string();
  result.metadata.xclbin_sha256 = sha256_file(options.xclbin);

  try {
    if (xrt::system::enumerate_devices() == 0)
      throw std::runtime_error("no XRT devices found");
    xrt::device device(0);
    try { result.metadata.device_bdf = device.get_info<xrt::info::device::bdf>(); } catch (...) {}
    try { result.metadata.device_name = device.get_info<xrt::info::device::name>(); } catch (...) {}
    const auto uuid = device.load_xclbin(options.xclbin);

    xrt::kernel kernel;
    try {
      kernel = xrt::kernel(device, uuid, kKernelName);
    } catch (...) {
      try {
        kernel = xrt::kernel(device, uuid, "llama3_attention_aie8_packet");
      } catch (...) {
        kernel = xrt::kernel(device, uuid, "llama3_aie8_v18_s32");
      }
    }

    const std::size_t q_count = q_elements(options.batch);
    const std::size_t kv_count = kv_elements(options.batch);
    const std::size_t q_bytes = q_count * sizeof(std::uint16_t);
    const std::size_t kv_bytes = kv_count * sizeof(std::uint16_t);
    xrt::bo q_bo(device, q_bytes, kernel.group_id(0));
    xrt::bo k_bo(device, kv_bytes, kernel.group_id(1));
    xrt::bo v_bo(device, kv_bytes, kernel.group_id(2));
    std::vector<xrt::bo> output_planes;
    output_planes.reserve(kOutputPlanes);
    if constexpr (kOutputPlanes == 1) {
      output_planes.emplace_back(device, q_bytes, kernel.group_id(3));
    } else {
      for (int plane = 0; plane < kOutputPlanes; ++plane)
        output_planes.emplace_back(device, q_bytes / 4,
                                   kernel.group_id(3 + plane));
    }
    auto* q_data = q_bo.map<std::uint16_t*>();
    auto* k_data = k_bo.map<std::uint16_t*>();
    auto* v_data = v_bo.map<std::uint16_t*>();
    std::mt19937 rng(options.seed);
    fill_random(q_data, q_count, rng);
    fill_random(k_data, kv_count, rng);
    fill_random(v_data, kv_count, rng);
    const std::vector<float> expected = options.verify
        ? reference_attention(q_data, k_data, v_data, options.batch)
        : std::vector<float>{};

    const auto sync_input = [&]() {
      q_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      k_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      v_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    };
    const auto sync_output = [&]() {
      for (auto& plane : output_planes) plane.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    };
    const auto launch = [&]() {
#if LLAMA3_OUTPUT_PLANES == 1
      return kernel(q_bo, k_bo, v_bo, output_planes[0], options.batch,
                    kSequenceLength);
#elif LLAMA3_OUTPUT_PLANES == 4
      return kernel(q_bo, k_bo, v_bo, output_planes[0], output_planes[1],
                    output_planes[2], output_planes[3], options.batch,
                    kSequenceLength);
#else
#error "Unsupported LLAMA3_OUTPUT_PLANES"
#endif
    };

    for (int run = 0; run < options.warmup; ++run) {
      sync_input();
      auto execution = launch();
      execution.wait();
      sync_output();
    }

    result.samples.h2d.reserve(options.runs);
    result.samples.kernel.reserve(options.runs);
    result.samples.d2h.reserve(options.runs);
    result.samples.e2e.reserve(options.runs);
    for (int run = 0; run < options.runs; ++run) {
      const auto e2e_start = std::chrono::steady_clock::now();
      sync_input();
      const auto h2d_stop = std::chrono::steady_clock::now();
      auto execution = launch();
      execution.wait();
      const auto kernel_stop = std::chrono::steady_clock::now();
      sync_output();
      const auto e2e_stop = std::chrono::steady_clock::now();
      const double h2d = std::chrono::duration<double, std::milli>(h2d_stop - e2e_start).count();
      const double kernel_time = std::chrono::duration<double, std::milli>(kernel_stop - h2d_stop).count();
      const double d2h = std::chrono::duration<double, std::milli>(e2e_stop - kernel_stop).count();
      const double e2e = std::chrono::duration<double, std::milli>(e2e_stop - e2e_start).count();
      result.samples.h2d.push_back(h2d);
      result.samples.kernel.push_back(kernel_time);
      result.samples.d2h.push_back(d2h);
      result.samples.e2e.push_back(e2e);
      if (options.profile && !options.quiet) {
        std::cout << "run=" << (run + 1) << " h2d_ms=" << std::fixed << std::setprecision(6)
                  << h2d << " kernel_ms=" << kernel_time << " d2h_ms=" << d2h
                  << " e2e_ms=" << e2e << '\n';
      }
    }

    result.h2d = summarize(result.samples.h2d);
    result.kernel = summarize(result.samples.kernel);
    result.d2h = summarize(result.samples.d2h);
    result.e2e = summarize(result.samples.e2e);

    std::vector<std::uint16_t> contiguous_output(q_count);
    if constexpr (kOutputPlanes == 1) {
      const auto* output_data =
          output_planes[0].map<const std::uint16_t*>();
      std::copy(output_data, output_data + q_count, contiguous_output.begin());
    } else {
      for (int plane = 0; plane < kOutputPlanes; ++plane) {
        const auto* plane_data =
            output_planes[plane].map<const std::uint16_t*>();
        for (int batch = 0; batch < options.batch; ++batch) {
          for (int head = 0; head < kQueryHeads; ++head) {
            for (int row = 0; row < kSequenceLength; ++row) {
              const std::size_t plane_row =
                  (static_cast<std::size_t>(batch) * kQueryHeads + head) *
                      kSequenceLength + row;
              for (int dim = 0; dim < 32; ++dim) {
                contiguous_output[
                    q_offset(batch, head, row, plane * 32 + dim)] =
                    plane_data[plane_row * 32 + dim];
              }
            }
          }
        }
      }
    }
    result.accuracy = evaluate_accuracy(contiguous_output.data(), expected, options.batch);
  } catch (const std::exception& error) {
    throw HardwareError(error.what());
  }

  result.attention_flops = 2.0 * kQueryHeads * kHeadDim * options.batch *
                           kSequenceLength * (kSequenceLength + 1);
  result.kernel_gflops = result.attention_flops / (result.kernel.p50 * 1.0e6);
  result.e2e_gflops = result.attention_flops / (result.e2e.p50 * 1.0e6);
  result.requests_per_second = options.batch * 1000.0 / result.e2e.p50;
  result.tokens_per_second = options.batch * kSequenceLength * 1000.0 / result.e2e.p50;
  return result;
}

std::string status_text(const std::string& value, const char* color, bool colors) {
  if (!colors) return value;
  return std::string(color) + value + "\033[0m";
}

void print_field(const std::string& label, const std::string& value) {
  std::cout << "  " << std::left << std::setw(23) << label << ": " << value << '\n';
}

std::string decimal(double value, int precision = 6) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

void print_summary(const Options& options, const BenchmarkResult& result) {
  const bool colors = isatty(fileno(stdout)) && std::getenv("NO_COLOR") == nullptr;
  constexpr const char* green = "\033[32m";
  constexpr const char* red = "\033[31m";
  constexpr const char* yellow = "\033[33m";
  const Accuracy& accuracy = result.accuracy;
  std::cout << "============================================================\n"
            << " AIE-FlashStream - Llama3-8B GQA Attention Accelerator\n"
            << "============================================================\n\n[Hardware]\n";
  print_field("Platform", "AMD Versal VCK5000");
  print_field("PL Frequency", std::to_string(kPlFrequencyMhz) + " MHz");
  print_field("AIE Compute Tiles", std::to_string(kAieTiles) + " / " +
                                       std::to_string(kAieTileBudget));
  print_field("Kernel", kKernelName);
  print_field("Device BDF", result.metadata.device_bdf);
  print_field("Device Name", result.metadata.device_name);
  print_field("XRT Version", result.metadata.xrt_version);
  std::cout << "\n[Workload]\n";
  print_field("Batch Size", std::to_string(options.batch));
  print_field("Sequence Length", std::to_string(kSequenceLength));
  print_field("Query / KV Heads", "32 / 8 (GQA 4:1)");
  print_field("Head Dimension", std::to_string(kHeadDim));
  print_field("Input / Output", "BF16 / BF16");
  print_field("QK Compute", LLAMA3_QK_COMPUTE_LABEL);
  print_field("Softmax / PV", "FP32");
  std::cout << "\n[Execution]\n";
  print_field("Warmup Runs", std::to_string(options.warmup));
  print_field("Measured Runs", std::to_string(options.runs));
  print_field("Random Seed", std::to_string(options.seed));
  std::cout << "\n[Latency]\n";
  print_field("H2D p50", decimal(result.h2d.p50) + " ms");
  print_field("Kernel p50", decimal(result.kernel.p50) + " ms");
  print_field("Kernel p95", decimal(result.kernel.p95) + " ms");
  print_field("D2H p50", decimal(result.d2h.p50) + " ms");
  print_field("End-to-End p50", decimal(result.e2e.p50) + " ms");
  print_field("End-to-End p95", decimal(result.e2e.p95) + " ms");
  print_field("Kernel / Request", decimal(result.kernel.p50 / options.batch) + " ms");
  print_field("E2E / Request", decimal(result.e2e.p50 / options.batch) + " ms");
  std::cout << "\n[Throughput]\n";
  print_field("Kernel Throughput", decimal(result.kernel_gflops) + " GFLOPS");
  print_field("E2E Throughput", decimal(result.e2e_gflops) + " GFLOPS");
  print_field("Requests / Second", decimal(result.requests_per_second, 3));
  print_field("Tokens / Second", decimal(result.tokens_per_second, 3));
  std::cout << "\n[Accuracy]\n";
  if (accuracy.measured) {
    print_field("Mean Abs Error", decimal(accuracy.mean_abs_error));
    print_field("P99 Abs Error", decimal(accuracy.p99_abs_error));
    print_field("P99.9 Abs Error", decimal(accuracy.p999_abs_error));
    print_field("Max Abs Error", decimal(accuracy.max_abs_error));
    print_field("Worst Coordinate", "batch=" + std::to_string(accuracy.worst_batch) +
        " head=" + std::to_string(accuracy.worst_head) +
        " row=" + std::to_string(accuracy.worst_row) +
        " dim=" + std::to_string(accuracy.worst_dimension));
    print_field("Reference / Hardware", decimal(accuracy.reference_value) + " / " +
        decimal(accuracy.hardware_value));
  } else {
    print_field("Accuracy Metrics", "NOT MEASURED (--no-verify)");
  }
  std::cout << "\n[Verification]\n";
  if (!accuracy.measured) {
    print_field("Functional Check", status_text("NOT VERIFIED", yellow, colors));
    print_field("Internal 0.003 Limit", status_text("NOT MEASURED", yellow, colors));
    print_field("Final Status", status_text("HARDWARE EXECUTED - NOT VERIFIED", yellow, colors));
  } else {
    print_field("Functional Check", status_text(accuracy.functional_pass ? "PASS" : "FAIL",
                                                accuracy.functional_pass ? green : red, colors));
    print_field("Internal 0.003 Limit", status_text(accuracy.internal_003_pass ? "PASS" : "FAIL",
                                                   accuracy.internal_003_pass ? green : red, colors));
    const std::string final = accuracy.functional_pass
        ? (accuracy.internal_003_pass ? "FUNCTIONALLY CORRECT" : "FUNCTIONALLY CORRECT (WARNING)")
        : "FUNCTIONAL FAILURE";
    print_field("Final Status", status_text(final,
        !accuracy.functional_pass ? red : (accuracy.internal_003_pass ? green : yellow), colors));
  }
  std::cout << "\n[Build]\n";
  print_field("Design / Host", std::string(kDesignVersion) + " / " + kHostVersion);
  print_field("Host Build Time", result.metadata.build_time);
  print_field("xclbin", result.metadata.xclbin_path);
  print_field("xclbin SHA256", result.metadata.xclbin_sha256);
  std::cout << "\n============================================================\n";
}

void write_statistics_json(std::ostream& out, const Statistics& stats, int indent) {
  const std::string pad(indent, ' ');
  out << "{\n" << pad << "  \"min\": " << stats.minimum
      << ",\n" << pad << "  \"mean\": " << stats.mean
      << ",\n" << pad << "  \"p50\": " << stats.p50
      << ",\n" << pad << "  \"p90\": " << stats.p90
      << ",\n" << pad << "  \"p95\": " << stats.p95
      << ",\n" << pad << "  \"p99\": " << stats.p99
      << ",\n" << pad << "  \"max\": " << stats.maximum
      << ",\n" << pad << "  \"stddev\": " << stats.standard_deviation
      << "\n" << pad << "}";
}

void write_json(const std::string& path, const Options& options,
                const BenchmarkResult& result) {
  ensure_parent(path);
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open JSON output: " + path);
  out << std::fixed << std::setprecision(9)
      << "{\n  \"schema_version\": 1,\n"
      << "  \"design\": \"" << kDesignVersion << "\",\n"
      << "  \"host_version\": \"" << kHostVersion << "\",\n"
      << "  \"platform\": \"VCK5000\",\n"
      << "  \"kernel\": \"" << kKernelName << "\",\n"
      << "  \"batch\": " << options.batch << ",\n"
      << "  \"sequence_length\": " << kSequenceLength << ",\n"
      << "  \"query_heads\": " << kQueryHeads << ",\n"
      << "  \"kv_heads\": " << kKvHeads << ",\n"
      << "  \"head_dimension\": " << kHeadDim << ",\n"
      << "  \"aie_tiles\": " << kAieTiles << ",\n"
      << "  \"output_planes\": " << kOutputPlanes << ",\n"
      << "  \"runs\": " << options.runs << ",\n"
      << "  \"warmup\": " << options.warmup << ",\n"
      << "  \"seed\": " << options.seed << ",\n"
      << "  \"timing_scope\": \"H2D + kernel + D2H\",\n"
      << "  \"percentile_method\": \"nearest-rank: sorted[ceil(q*N)-1]\",\n"
      << "  \"timing_ms\": {\n    \"h2d\": ";
  write_statistics_json(out, result.h2d, 4);
  out << ",\n    \"kernel\": "; write_statistics_json(out, result.kernel, 4);
  out << ",\n    \"d2h\": "; write_statistics_json(out, result.d2h, 4);
  out << ",\n    \"e2e\": "; write_statistics_json(out, result.e2e, 4);
  out << "\n  },\n  \"per_request_ms\": {\n"
      << "    \"kernel_p50\": " << result.kernel.p50 / options.batch << ",\n"
      << "    \"e2e_p50\": " << result.e2e.p50 / options.batch << "\n  },\n"
      << "  \"throughput\": {\n"
      << "    \"attention_flops_per_batch\": " << result.attention_flops << ",\n"
      << "    \"kernel_gflops\": " << result.kernel_gflops << ",\n"
      << "    \"e2e_gflops\": " << result.e2e_gflops << ",\n"
      << "    \"requests_per_second\": " << result.requests_per_second << ",\n"
      << "    \"tokens_per_second\": " << result.tokens_per_second << "\n  },\n"
      << "  \"accuracy\": {\n"
      << "    \"measured\": " << (result.accuracy.measured ? "true" : "false") << ",\n"
      << "    \"mean_abs_error\": " << result.accuracy.mean_abs_error << ",\n"
      << "    \"p99_abs_error\": " << result.accuracy.p99_abs_error << ",\n"
      << "    \"p999_abs_error\": " << result.accuracy.p999_abs_error << ",\n"
      << "    \"max_abs_error\": " << result.accuracy.max_abs_error << ",\n"
      << "    \"functional_pass\": " << (result.accuracy.functional_pass ? "true" : "false") << ",\n"
      << "    \"internal_003_pass\": " << (result.accuracy.internal_003_pass ? "true" : "false") << ",\n"
      << "    \"worst_coordinate\": {\"batch\": " << result.accuracy.worst_batch
      << ", \"head\": " << result.accuracy.worst_head
      << ", \"row\": " << result.accuracy.worst_row
      << ", \"dimension\": " << result.accuracy.worst_dimension << "},\n"
      << "    \"reference_value\": " << result.accuracy.reference_value << ",\n"
      << "    \"hardware_value\": " << result.accuracy.hardware_value << "\n  },\n"
      << "  \"environment\": {\n"
      << "    \"xclbin_path\": \"" << json_escape(result.metadata.xclbin_path) << "\",\n"
      << "    \"xclbin_sha256\": \"" << result.metadata.xclbin_sha256 << "\",\n"
      << "    \"host_build_time\": \"" << json_escape(result.metadata.build_time) << "\",\n"
      << "    \"xrt_version\": \"" << json_escape(result.metadata.xrt_version) << "\",\n"
      << "    \"device_bdf\": \"" << json_escape(result.metadata.device_bdf) << "\",\n"
      << "    \"device_name\": \"" << json_escape(result.metadata.device_name) << "\"\n  }\n}\n";
  if (!out) throw std::runtime_error("failed while writing JSON output: " + path);
}

void write_csv(const std::string& path, const BenchmarkResult& result) {
  ensure_parent(path);
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open CSV output: " + path);
  out << "run,h2d_ms,kernel_ms,d2h_ms,e2e_ms\n" << std::fixed << std::setprecision(9);
  for (std::size_t index = 0; index < result.samples.e2e.size(); ++index) {
    out << (index + 1) << ',' << result.samples.h2d[index] << ','
        << result.samples.kernel[index] << ',' << result.samples.d2h[index]
        << ',' << result.samples.e2e[index] << '\n';
  }
  if (!out) throw std::runtime_error("failed while writing CSV output: " + path);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = parse_options(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "Argument error: " << error.what() << "\n\n" << usage(argv[0]);
    return 1;
  }

  BenchmarkResult result;
  try {
    result = run_benchmark(options);
  } catch (const HardwareError& error) {
    std::cerr << "XRT/device/kernel error: " << error.what() << '\n';
    return 2;
  }

  try {
    if (!options.output_json.empty()) write_json(options.output_json, options, result);
    if (!options.output_csv.empty()) write_csv(options.output_csv, result);
  } catch (const std::exception& error) {
    std::cerr << "Output file error: " << error.what() << '\n';
    return 1;
  }

  print_summary(options, result);
  if (result.accuracy.measured && !result.accuracy.functional_pass) return 3;
  return 0;
}
