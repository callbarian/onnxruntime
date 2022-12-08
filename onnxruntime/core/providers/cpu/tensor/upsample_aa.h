// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/upsample.h"
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif
#include "core/providers/cpu/tensor/upsamplebase.h"
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {

struct FilterParamsBaseAA {
  std::vector<int64_t> bound;
  std::vector<float> original;
  int64_t window_size = 2;
  BufferUniquePtr weight_coefficients;
};

struct FilterParamsAA {
  float support_size = 2.0f;
  float cubic_coeff_a = -0.75f;

  /* Handles values form -640 to 639. */
  uint8_t* clip8_lookups_table;

  FilterParamsBaseAA dim_x;
  FilterParamsBaseAA dim_y;
  FilterParamsBaseAA dim_z;

  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L20-L29

  static int32_t round_up(float f) {
    return ((int32_t)((f) >= 0.0 ? (f) + 0.5F : (f)-0.5F));
  }

  void init_clip_lookup() {
    if (clip8_lookups_table[1279] == 255) {
      return;
    }
    for (int i = 0; i < 1280; ++i) {
      clip8_lookups_table[i] = static_cast<uint8_t>(std::min(std::max(i - 640, 0), 255));
    }
  }
  virtual float filter(float x) const = 0;
};

struct BilinearParamsAA : FilterParamsAA {
  float filter(float x) const override {
    if (x < 0.0) {
      x = -x;
    }
    if (x < 1.0) {
      return 1.0 - x;
    }
    return 0.0;
  }
};

struct BiCubicParamsAA : FilterParamsAA {
  BiCubicParamsAA() {
    support_size = (4.0f);
  }
  float filter(float x) const override {
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
     */
    if (x < 0.0) {
      x = -x;
    }
    if (x < 1.0) {
      return ((cubic_coeff_a + 2.0) * x - (cubic_coeff_a + 3.0)) * x * x + 1;
    }
    if (x < 2.0) {
      return (((x - 5) * x + 8) * x - 4) * cubic_coeff_a;
    }
    return 0.0;
  }
};

struct TriLinearParamsAA : FilterParamsAA {
  float filter(float x) const override {
    if (x < 0.0) {
      x = -x;
    }
    if (x < 1.0) {
      return 1.0 - x;
    }
    return 0.0;
  }
};

template <typename T>
struct AccumulateType {
  using type = int32_t;
  using Dtype = T;
};

template <>
struct AccumulateType<float> {
  using type = float;
};

template <>
struct AccumulateType<double> {
  using type = double;
};

void SetupUpsampleFilterAA(FilterParamsAA& p,
                           const gsl::span<int64_t> input_h_w_c,
                           const gsl::span<int64_t> output_h_w_c,
                           const gsl::span<float> scale_h_w_c,
                           const std::vector<float>& roi,
                           AllocatorPtr& alloc,
                           const GetOriginalCoordinateFunc& get_original_coordinate,
                           const int32_t dtype, bool exclude_outside, const bool is_nchw);

template <typename T>
void UpsampleBaseAA(FilterParamsAA& p,
                    const int32_t batch_size,
                    const int32_t num_channels,
                    const int32_t input_height,
                    const int32_t input_width,
                    const int32_t output_height,
                    const int32_t output_width,
                    const bool use_extrapolation,
                    const float extrapolation_value,
                    const T* const XdataBase,
                    T* const YdataBase,
                    AllocatorPtr& alloc,
                    concurrency::ThreadPool* tp) {
  const uint8_t* clip8_lookups = &p.clip8_lookups_table[640];

  constexpr bool is_8bit_data =
      (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value);

  auto image_temp_buffer = BufferUniquePtr(alloc->Alloc(input_height *
                                                        output_width * num_channels * sizeof(T)),
                                           BufferDeleter(alloc));

  int32_t mag_factor = 1 << (22 - 1);
  using ACtype = typename AccumulateType<T>::type;

  for (int32_t n = 0; n < batch_size; ++n) {
    auto* temp_buffer = static_cast<T*>(image_temp_buffer.get());
    // horizon interpolate
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_channels,
        [&](std::ptrdiff_t c) {
          const T* const Xdata =
              XdataBase + (n * num_channels + static_cast<int32_t>(c)) *
                              (input_height * input_width);
          T* const Ydata = temp_buffer + (n * num_channels + static_cast<int32_t>(c)) *
                                             (input_height * output_width);
          for (int32_t y = 0; y < input_height; ++y) {
            for (int32_t x = 0; x < output_width; ++x) {
              const int32_t output_offset = output_width * y + x;
              // when use_extrapolation is set and original index of x or y is out of the dim range
              // then use extrapolation_value as the output value.
              if (use_extrapolation &&
                  ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                   (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
                Ydata[output_offset] = static_cast<T>(extrapolation_value);
                continue;
              }
              ACtype output = 0;
              if constexpr (is_8bit_data) {
                output = mag_factor;
              }
              const auto* weight_coeff =
                  reinterpret_cast<const ACtype*>(p.dim_x.weight_coefficients.get()) +
                  p.dim_x.window_size * x;
              int32_t xmin = p.dim_x.bound[x * 2];
              int32_t xmax = p.dim_x.bound[x * 2 + 1];
              for (; xmin < xmax; ++xmin) {
                output += Xdata[y * input_width + xmin] * (*weight_coeff++);
              }

              if constexpr (is_8bit_data) {
                Ydata[output_offset] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata[output_offset] = p.round_up(output);
              } else {
                Ydata[output_offset] = static_cast<T>(output);
              }
            }
          }
        });
    // vertical interpolate
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_channels,
        [&](std::ptrdiff_t c) {
          const T* const Xdata =
              temp_buffer + (n * num_channels + static_cast<int32_t>(c)) *
                                (input_height * output_width);
          T* const Ydata =
              YdataBase + (n * num_channels + static_cast<int32_t>(c)) *
                              (output_height * output_width);
          for (int32_t y = 0; y < output_height; ++y) {
            for (int32_t x = 0; x < output_width; ++x) {
              const int32_t output_offset = output_width * y + x;
              if (use_extrapolation &&
                  ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                   (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
                Ydata[output_offset] = static_cast<T>(extrapolation_value);
                continue;
              }

              ACtype output = 0;
              if constexpr (is_8bit_data) {
                output = mag_factor;
              }
              const auto* weight_coeff =
                  reinterpret_cast<const ACtype*>(p.dim_y.weight_coefficients.get()) +
                  p.dim_y.window_size * y;
              int32_t ymin = p.dim_y.bound[y * 2];
              int32_t ymax = p.dim_y.bound[y * 2 + 1];
              for (; ymin < ymax; ++ymin) {
                output +=
                    Xdata[ymin * output_width + x] * (*weight_coeff++);
              }
              if constexpr (is_8bit_data) {
                Ydata[output_offset] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata[output_offset] = p.round_up(output);
              } else {  // float double
                Ydata[output_offset] = static_cast<T>(output);
              }
            }
          }
        });
  }
}

template <typename T>
void UpsampleBilinearAA(const int32_t batch_size,
                        const int32_t num_channels,
                        const int32_t input_height,
                        const int32_t input_width,
                        const int32_t output_height,
                        const int32_t output_width,
                        const float height_scale,
                        const float width_scale,
                        const std::vector<float>& roi,
                        const bool use_extrapolation,
                        const float extrapolation_value,
                        bool exclude_outside,
                        const Tensor* const X,
                        T* const YdataBase,
                        AllocatorPtr& alloc,
                        const GetOriginalCoordinateFunc& get_original_coordinate,
                        concurrency::ThreadPool* tp) {
  const auto* XdataBase = X->Data<T>();

  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BilinearParamsAA p;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), exclude_outside, true);
  return UpsampleBaseAA<T>(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                           use_extrapolation, extrapolation_value,
                           XdataBase, YdataBase, alloc, tp);
}

template <typename T>
void NhwcUpsampleBilinearAA(const int32_t batch_size,
                            const int32_t num_channels,
                            const int32_t input_height,
                            const int32_t input_width,
                            const int32_t output_height,
                            const int32_t output_width,
                            const float height_scale,
                            const float width_scale,
                            const std::vector<float>& roi,
                            const bool use_extrapolation,
                            const float extrapolation_value,
                            bool exclude_outside,
                            const Tensor* const X,
                            T* const YdataBase,
                            AllocatorPtr& alloc,
                            const GetOriginalCoordinateFunc& get_original_coordinate,
                            concurrency::ThreadPool* tp) {
  const auto* XdataBase = X->Data<T>();

  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BilinearParamsAA p;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), exclude_outside, false);
  return NhwcUpsampleBasicAA(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                             use_extrapolation, extrapolation_value,
                             XdataBase, YdataBase, alloc, tp);
}

template <typename T>
void NhwcResizeBiCubicAA(const int32_t batch_size,
                         const int32_t num_channels,
                         const int32_t input_height,
                         const int32_t input_width,
                         const int32_t output_height,
                         const int32_t output_width,
                         const float height_scale,
                         const float width_scale,
                         float cubic_coeff_a,
                         bool use_extrapolation,
                         float extrapolation_value,
                         bool exclude_outside,
                         const std::vector<float>& roi,
                         const Tensor* const X,
                         T* const YdataBase,
                         AllocatorPtr& alloc,
                         const GetOriginalCoordinateFunc& get_original_coordinate,
                         concurrency::ThreadPool* tp) {
  const auto* XdataBase = X->Data<T>();

  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BiCubicParamsAA p;
  p.cubic_coeff_a = cubic_coeff_a;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), exclude_outside, false);
  return NhwcUpsampleBasicAA(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                             use_extrapolation, extrapolation_value,
                             XdataBase, YdataBase, alloc, tp);
}

template <typename T>
void NhwcUpsampleBasicAA(FilterParamsAA& p,
                         const int32_t batch_size,
                         const int32_t num_channels,
                         const int32_t input_height,
                         const int32_t input_width,
                         const int32_t output_height,
                         const int32_t output_width,
                         const bool use_extrapolation,
                         const float extrapolation_value,
                         const T* const XdataBase,
                         T* const YdataBase,
                         AllocatorPtr& alloc,
                         concurrency::ThreadPool* tp) {
  const uint8_t* clip8_lookups = &p.clip8_lookups_table[640];

  constexpr bool is_8bit_data =
      (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value);

  auto image_temp_buffer = BufferUniquePtr(alloc->Alloc(input_height *
                                                        output_width * num_channels * sizeof(T)),
                                           BufferDeleter(alloc));

  using ACtype = typename AccumulateType<T>::type;
  int32_t mag_factor = 1 << (22 - 1);

  for (int32_t n = 0; n < batch_size; ++n) {
    auto* temp_buffer = static_cast<T*>(image_temp_buffer.get());

    // horizon interpolate
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(input_height) * output_width,
        static_cast<double>(num_channels * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          const T* const Xdata =
              XdataBase +
              n * (input_height * input_width) * num_channels;
          T* const Ydata =
              temp_buffer + n * (input_height * output_width) * num_channels;
          for (std::ptrdiff_t i = first; i < last; ++i) {
            const int32_t x = static_cast<int32_t>(i % output_width);
            const int32_t y = static_cast<int32_t>(i / output_width);
            T* const Ydata_with_offset = Ydata + (output_width * y + x) * num_channels;
            if (use_extrapolation && ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                                      (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
              for (int32_t c = 0; c < num_channels; ++c) {
                Ydata_with_offset[c] = static_cast<T>(extrapolation_value);
              }
              continue;
            }

            const auto* weight_coeff =
                reinterpret_cast<const ACtype*>(p.dim_x.weight_coefficients.get()) +
                p.dim_x.window_size * x;
            int32_t xmin = p.dim_x.bound[x * 2];
            int32_t xmax = p.dim_x.bound[x * 2 + 1];
            for (int32_t c = 0; c < num_channels; ++c) {
              const auto* weight_coeff_start = weight_coeff;
              ACtype output = 0;
              if constexpr (is_8bit_data) {
                output = mag_factor;
              }
              for (int idx = xmin; idx < xmax; ++idx) {
                // printf("%f*%f + ", *weight_coeff_start, Xdata[(y * input_width + idx) * num_channels + c]);

                output += Xdata[(y * input_width + idx) * num_channels + c] *
                          (*weight_coeff_start++);
              }
              // printf("=%f\n", output);
              if constexpr (is_8bit_data) {
                Ydata_with_offset[c] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata_with_offset[c] = p.round_up(output);
              } else {  // float double
                Ydata_with_offset[c] = output;
              }
            }
          }
        });

    // vertical interpolate
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(output_height) * output_width,
        static_cast<double>(num_channels * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          const T* const Xdata = temp_buffer + n * (input_height * output_width) * num_channels;
          T* const Ydata = YdataBase + n * (output_height * output_width) * num_channels;

          for (std::ptrdiff_t i = first; i < last; ++i) {
            const int32_t x = static_cast<int32_t>(i % output_width);
            const int32_t y = static_cast<int32_t>(i / output_width);
            T* const Ydata_with_offset = Ydata + (output_width * y + x) * num_channels;

            if (use_extrapolation && ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                                      (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
              for (int32_t c = 0; c < num_channels; ++c) {
                Ydata_with_offset[c] = static_cast<T>(extrapolation_value);
              }
              continue;
            }

            const auto* weight_coeff =
                reinterpret_cast<const ACtype*>(p.dim_y.weight_coefficients.get()) + p.dim_y.window_size * y;
            int32_t ymin = p.dim_y.bound[y * 2];
            int32_t ymax = p.dim_y.bound[y * 2 + 1];

            for (int32_t c = 0; c < num_channels; ++c) {
              const auto* weight_coeff_start = weight_coeff;
              ACtype output = 0;
              if constexpr (is_8bit_data) {
                output = mag_factor;
              }
              for (int idy = ymin; idy < ymax; ++idy) {
                output += Xdata[(idy * output_width + x) * num_channels + c] *
                          (*weight_coeff_start++);
              }
              if constexpr (is_8bit_data) {
                Ydata_with_offset[c] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata_with_offset[c] = p.round_up(output);
              } else {  // float double
                Ydata_with_offset[c] = output;
              }
            }
          }
        });
  }
}

template <typename T>
inline void InterpolateCompute(const T* Xdata, const int32_t stride,
                               const float* weight_coeff,
                               const int64_t* idx_bound, T* Ydata) {
  float output = 0;

  for (int32_t idx = idx_bound[0]; idx < idx_bound[1]; ++idx) {
    output += Xdata[idx * stride] * (*weight_coeff++);
  }
  *Ydata = static_cast<T>(output);
}

template <typename T>
inline void InterpolateLoopForSingleDim(
    const T* Xdata, int32_t dim_size, int32_t section_idx,
    const int32_t y_stride, const int32_t x_stride, const float* weight_coeff,
    const std::vector<int64_t>& idx_bound, int32_t window_size, T* Ydata) {
  for (int32_t step = 0; step < dim_size; ++step) {
    InterpolateCompute(
        Xdata, y_stride,
        &weight_coeff[(x_stride == 0) ? window_size * step
                                      : window_size * section_idx],
        &idx_bound[(x_stride == 0) ? step * 2 : section_idx * 2], Ydata);
    Ydata++;
    Xdata += x_stride;
  }
}

template <typename T>
void LoopInDimN(const T* Xdata_base, size_t dim_idx, size_t section_idx,
                const std::vector<int32_t>& input_stride,
                const std::vector<int32_t>& output_stride, const size_t compute_dim,
                const float* weight_coeff, const std::vector<int64_t>& idx_bound,
                int32_t window_size, T* Ydata_base) {
  const int32_t x_ofs =
      section_idx * ((compute_dim == dim_idx) ? 0 : input_stride[dim_idx]);

  const T* const Xdata = Xdata_base + x_ofs;
  T* const Ydata = Ydata_base + section_idx * output_stride[dim_idx];
  if (dim_idx < input_stride.size() - 2) {
    for (int32_t sub_sec_idx = 0;
         sub_sec_idx < output_stride[dim_idx] / output_stride[dim_idx + 1];
         ++sub_sec_idx) {
      LoopInDimN(Xdata, dim_idx + 1, sub_sec_idx, input_stride, output_stride,
                 compute_dim, weight_coeff, idx_bound, window_size, Ydata);
    }
    return;
  }
  int32_t x_stride = compute_dim == output_stride.size() - 1 ? 0 : output_stride[compute_dim + 1];
  InterpolateLoopForSingleDim(
      Xdata, output_stride[dim_idx] / output_stride[dim_idx + 1], section_idx,
      output_stride[compute_dim], x_stride, weight_coeff, idx_bound,
      window_size, Ydata);
}

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 6001)
#endif
template <typename T>
void ResizeBiCubicAA(int64_t batch_size,
                     int64_t num_channels,
                     int64_t input_height,
                     int64_t input_width,
                     int64_t output_height,
                     int64_t output_width,
                     float height_scale,
                     float width_scale,
                     float cubic_coeff_a,
                     bool use_extrapolation,
                     float extrapolation_value,
                     bool exclude_outside,
                     const std::vector<float>& roi,
                     const Tensor* X,
                     T* YdataBase,
                     AllocatorPtr& alloc,
                     const GetOriginalCoordinateFunc& get_original_coordinate,
                     concurrency::ThreadPool* tp) {
  const auto* XdataBase = X->Data<T>();
  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BiCubicParamsAA p;
  p.cubic_coeff_a = cubic_coeff_a;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), exclude_outside, false);

  return UpsampleBaseAA<T>(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                           use_extrapolation, extrapolation_value,
                           XdataBase, YdataBase, alloc, tp);
}

/*
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
template <typename T>
void UpsampleTrilinear(int64_t batch_size,
                       int64_t num_channels,
                       int64_t input_depth,
                       int64_t input_height,
                       int64_t input_width,
                       int64_t output_depth,
                       int64_t output_height,
                       int64_t output_width,
                       float depth_scale,
                       float height_scale,
                       float width_scale,
                       const std::vector<float>& roi,
                       bool use_extrapolation,
                       float extrapolation_value,
                       const Tensor* X,
                       T* YdataBase,
                       AllocatorPtr& alloc,
                       const GetOriginalCoordinateFunc& get_original_coordinate,
                       concurrency::ThreadPool* tp) {
  const auto* XdataBase = X->Data<T>();
  int64_t input_paras[] = {input_height, input_width, input_depth};
  int64_t output_paras[] = {output_height, output_width, output_depth};
  float scale_paras[] = {height_scale, width_scale, depth_scale};
  TriLinearParamsAA p;
  SetupUpsampleFilterAA(p, input_paras, output_paras, scale_paras, roi,
                        alloc, get_original_coordinate, X->GetElementType(), true, false);

  for (int64_t n = 0; n < batch_size; ++n) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, static_cast<std::ptrdiff_t>(num_channels),
        [&](std::ptrdiff_t c) {
          const T* Xdata = XdataBase + (n * num_channels + c) * (input_depth * input_height * input_width);
          T* Ydata = YdataBase + (n * num_channels + c) * (output_depth * output_height * output_width);
          for (int64_t z = 0; z < output_depth; ++z) {
            for (int64_t y = 0; y < output_height; ++y) {
              for (int64_t x = 0; x < output_width; ++x) {
                // when use_extrapolation is set and original index of x or y is out of the dim range
                // then use extrapolation_value as the output value.
                if (use_extrapolation &&
                    ((p.z_original[narrow<size_t>(z)] < 0 || p.z_original[narrow<size_t>(z)] > static_cast<float>(input_depth - 1)) ||
                     (p.dim_y.original[narrow<size_t>(y)] < 0 || p.dim_y.original[narrow<size_t>(y)] > static_cast<float>(input_height - 1)) ||
                     (p.dim_x.original[narrow<size_t>(x)] < 0 || p.dim_x.original[narrow<size_t>(x)] > static_cast<float>(input_width - 1)))) {
                  Ydata[output_width * output_height * z + output_width * y + x] =
                      static_cast<T>(extrapolation_value);
                  continue;
                }

                // subscript ordering in the variable - (xyz)
                T X111 = Xdata[p.input_height_width_mul_z1[narrow<size_t>(z)] + p.input_width_mul_y1[narrow<size_t>(y)] + p.in_x1[narrow<size_t>(x)]];
                T X211 = Xdata[p.input_height_width_mul_z1[narrow<size_t>(z)] + p.input_width_mul_y1[narrow<size_t>(y)] + p.in_x2[narrow<size_t>(x)]];
                T X121 = Xdata[p.input_height_width_mul_z1[narrow<size_t>(z)] + p.input_width_mul_y2[narrow<size_t>(y)] + p.in_x1[narrow<size_t>(x)]];
                T X221 = Xdata[p.input_height_width_mul_z1[narrow<size_t>(z)] + p.input_width_mul_y2[narrow<size_t>(y)] + p.in_x2[narrow<size_t>(x)]];

                T X112 = Xdata[p.input_height_width_mul_z2[narrow<size_t>(z)] + p.input_width_mul_y1[narrow<size_t>(y)] + p.in_x1[narrow<size_t>(x)]];
                T X212 = Xdata[p.input_height_width_mul_z2[narrow<size_t>(z)] + p.input_width_mul_y1[narrow<size_t>(y)] + p.in_x2[narrow<size_t>(x)]];
                T X122 = Xdata[p.input_height_width_mul_z2[narrow<size_t>(z)] + p.input_width_mul_y2[narrow<size_t>(y)] + p.in_x1[narrow<size_t>(x)]];
                T X222 = Xdata[p.input_height_width_mul_z2[narrow<size_t>(z)] + p.input_width_mul_y2[narrow<size_t>(y)] + p.in_x2[narrow<size_t>(x)]];

                Ydata[output_width * output_height * z + output_width * y + x] =
                    static_cast<T>(p.dx2[narrow<size_t>(x)] * p.dy2[narrow<size_t>(y)] * p.dz2[narrow<size_t>(z)] * X111 +
                                   p.dx1[narrow<size_t>(x)] * p.dy2[narrow<size_t>(y)] * p.dz2[narrow<size_t>(z)] * X211 +
                                   p.dx2[narrow<size_t>(x)] * p.dy1[narrow<size_t>(y)] * p.dz2[narrow<size_t>(z)] * X121 +
                                   p.dx1[narrow<size_t>(x)] * p.dy1[narrow<size_t>(y)] * p.dz2[narrow<size_t>(z)] * X221 +

                                   p.dx2[narrow<size_t>(x)] * p.dy2[narrow<size_t>(y)] * p.dz1[narrow<size_t>(z)] * X112 +
                                   p.dx1[narrow<size_t>(x)] * p.dy2[narrow<size_t>(y)] * p.dz1[narrow<size_t>(z)] * X212 +
                                   p.dx2[narrow<size_t>(x)] * p.dy1[narrow<size_t>(y)] * p.dz1[narrow<size_t>(z)] * X122 +
                                   p.dx1[narrow<size_t>(x)] * p.dy1[narrow<size_t>(y)] * p.dz1[narrow<size_t>(z)] * X222);
              }
            }
          }
          Xdata += input_depth * input_height * input_width;
          Ydata += output_depth * output_width * output_height;
        });
  }
}
*/

}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
