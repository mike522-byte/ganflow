# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for resampling images."""

import tensorflow as tf


def safe_gather_nd(params, indices):
  """Gather slices from params into a Tensor with shape specified by indices.

  Similar functionality to tf.gather_nd with difference: when index is out of
  bound, always return 0.

  Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
      tensor.

  Returns:
    A Tensor. Has the same type as params. Values from params gathered from
    specified indices (if they exist) otherwise zeros, with shape
    indices.shape[:-1] + params.shape[indices.shape[-1]:].
  """
  params_shape = tf.shape(params)     # 形状
  indices_shape = tf.shape(indices)
  slice_dimensions = indices_shape[-1]

  max_index = params_shape[:slice_dimensions] - 1
  min_index = tf.zeros_like(max_index, dtype=tf.int32) #返回一个全部为0的张量，大小和输入的一样

  clipped_indices = tf.clip_by_value(indices, min_index, max_index) #可以将一个张量中的数值限制在一个范围之内

  # Check whether each component of each index is in range [min, max], and
  # allow an index only if all components are in range:
  mask = tf.reduce_all(           #计算一个张量在维度上元素的“逻辑和”
      tf.logical_and(indices >= min_index, indices <= max_index), -1)
  mask = tf.expand_dims(mask, -1)
#对dtype 执行一个安全的饱和cast的value.
  return (tf.cast(mask, dtype=params.dtype) *
          tf.gather_nd(params, clipped_indices))
#将参数中的切片收集到由索引指定的形状的张量中.

def resampler(data, warp, name='resampler'):
  """Resamples input data at user defined coordinates.

  Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
      data_num_channels]` containing 2D data that will be resampled.
    warp: Tensor shape `[batch_size, dim_0, ... , dim_n, 2]` containing the
      coordinates at which resampling will be performed.
    name: Optional name of the op.

  Returns:
    Tensor of resampled values from `data`. The output tensor shape is
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.
  """
  data = tf.convert_to_tensor(data)
  warp = tf.convert_to_tensor(warp)
  with tf.name_scope(name + '/unstack_warp'):#返回在定义 Python 操作时使用的上下文管理器
    warp_x, warp_y = tf.unstack(warp, axis=-1)#将秩为 R 的张量的给定维度出栈为秩为 (R-1) 的张量.
  return resampler_with_unstacked_warp(data, warp_x, warp_y, name=name)


def resampler_with_unstacked_warp(data,
                                  warp_x,
                                  warp_y,
                                  safe=True,
                                  name='resampler'):
  """Resamples input data at user defined coordinates.

  The resampler functions in the same way as `resampler` above, with the
  following differences:
  1. The warp coordinates for x and y are given as separate tensors.
  2. If warp_x and warp_y are known to be within their allowed bounds, (that is,
     0 <= warp_x <= width_of_data - 1, 0 <= warp_y <= height_of_data - 1) we
     can disable the `safe` flag.

  Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
      data_num_channels]` containing 2D data that will be resampled.
    warp_x: Tensor of shape `[batch_size, dim_0, ... , dim_n]` containing the x
      coordinates at which resampling will be performed.
    warp_y: Tensor of the same shape as warp_x containing the y coordinates at
      which resampling will be performed.
    safe: A boolean, if True, warp_x and warp_y will be clamped to their bounds.
      Disable only if you know they are within bounds, otherwise a runtime
      exception will be thrown.
    name: Optional name of the op.

  Returns:
     Tensor of resampled values from `data`. The output tensor shape is
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.

  Raises:
    ValueError: If warp_x, warp_y and data have incompatible shapes.
  """

  with tf.name_scope(name):
    warp_x = tf.convert_to_tensor(warp_x)
    warp_y = tf.convert_to_tensor(warp_y)
    data = tf.convert_to_tensor(data)
    if not warp_x.shape.is_compatible_with(warp_y.shape):
      raise ValueError(
          'warp_x and warp_y are of incompatible shapes: %s vs %s ' %
          (str(warp_x.shape), str(warp_y.shape)))
    warp_shape = tf.shape(warp_x)
    if warp_x.shape[0] != data.shape[0]:
      raise ValueError(
          '\'warp_x\' and \'data\' must have compatible first '
          'dimension (batch size), but their shapes are %s and %s ' %
          (str(warp_x.shape[0]), str(data.shape[0])))
    # Compute the four points closest to warp with integer value.
    warp_floor_x = tf.math.floor(warp_x)
    warp_floor_y = tf.math.floor(warp_y)
    # Compute the weight for each point.
    right_warp_weight = warp_x - warp_floor_x
    down_warp_weight = warp_y - warp_floor_y

    warp_floor_x = tf.cast(warp_floor_x, tf.int32)
    warp_floor_y = tf.cast(warp_floor_y, tf.int32)
    warp_ceil_x = tf.cast(tf.math.ceil(warp_x), tf.int32)
    warp_ceil_y = tf.cast(tf.math.ceil(warp_y), tf.int32)

    left_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, right_warp_weight.dtype), right_warp_weight)
    up_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, down_warp_weight.dtype), down_warp_weight)

    # Extend warps from [batch_size, dim_0, ... , dim_n, 2] to
    # [batch_size, dim_0, ... , dim_n, 3] with the first element in last
    # dimension being the batch index.

    # A shape like warp_shape but with all sizes except the first set to 1:
    warp_batch_shape = tf.concat(
        [warp_shape[0:1], tf.ones_like(warp_shape[1:])], 0)

    warp_batch = tf.reshape(
        tf.range(warp_shape[0], dtype=tf.int32), warp_batch_shape)

    # Broadcast to match shape:
    warp_batch += tf.zeros_like(warp_y, dtype=tf.int32)
    left_warp_weight = tf.expand_dims(left_warp_weight, axis=-1)
    down_warp_weight = tf.expand_dims(down_warp_weight, axis=-1)
    up_warp_weight = tf.expand_dims(up_warp_weight, axis=-1)
    right_warp_weight = tf.expand_dims(right_warp_weight, axis=-1)

    up_left_warp = tf.stack([warp_batch, warp_floor_y, warp_floor_x], axis=-1)
    up_right_warp = tf.stack([warp_batch, warp_floor_y, warp_ceil_x], axis=-1)
    down_left_warp = tf.stack([warp_batch, warp_ceil_y, warp_floor_x], axis=-1)
    down_right_warp = tf.stack([warp_batch, warp_ceil_y, warp_ceil_x], axis=-1)

    def gather_nd(params, indices):
      return (safe_gather_nd if safe else tf.gather_nd)(params, indices)

    # gather data then take weighted average to get resample result.
    result = (
        (gather_nd(data, up_left_warp) * left_warp_weight +
         gather_nd(data, up_right_warp) * right_warp_weight) * up_warp_weight +
        (gather_nd(data, down_left_warp) * left_warp_weight +
         gather_nd(data, down_right_warp) * right_warp_weight) *
        down_warp_weight)
    result_shape = (
        warp_x.get_shape().as_list() + data.get_shape().as_list()[-1:])
    result.set_shape(result_shape)
    return result
