
import math
import sys
import time
import gin
from absl import app



import math
# from absl import flags
import tensorflow as tf
import collections
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization

from uflow import uflow_flags
from uflow import uflow_model1
from uflow import uflow_utils
from uflow import uflow_plotting

FLAGS = tf.compat.v1.app.flags.FLAGS

def normalize_features(feature_list, normalize, center ,moments_across_channels,
                       moments_across_images):
    """Normalizes feature tensors (e.g., before computing the cost volume).
    Args:
      feature: tf.tensors, with dimensions [b, h, w, c]
      normalize: bool flag, divide features by their standard deviation
      center: bool flag, subtract feature mean
      moments_across_channels: bool flag, compute mean and std across channels
      moments_across_images: bool flag, compute mean and std across images

    Returns:
      list, normalized feature_list
    """

    # Compute feature statistics.

    statistics = collections.defaultdict(list)
    axes = [-3, -2, -1] if moments_across_channels else [-3, -2]
    for feature_image in feature_list:
        mean, variance = tf.nn.moments(x=feature_image, axes=axes, keepdims=True)
        statistics['mean'].append(mean)
        statistics['var'].append(variance)

    if moments_across_images:
        statistics['mean'] = ([tf.reduce_mean(input_tensor=statistics['mean'])] *
                              len(feature_list))
        statistics['var'] = [tf.reduce_mean(input_tensor=statistics['var'])
                             ] * len(feature_list)

    statistics['std'] = [tf.sqrt(v + 1e-16) for v in statistics['var']]

    # Center and normalize features.

    if center:
        feature_list = [
            f - mean for f, mean in zip(feature_list, statistics['mean'])
        ]
    if normalize:
        feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]

    return feature_list


def compute_cost_volume(features1, features2, max_displacement):
  """Compute the cost volume between features1 and features2.

  Displace features2 up to max_displacement in any direction and compute the
  per pixel cost of features1 and the displaced features2.

  Args:
    features1: tf.tensor of shape [b, h, w, c]
    features2: tf.tensor of shape [b, h, w, c]
    max_displacement: int, maximum displacement for cost volume computation.

  Returns:
    tf.tensor of shape [b, h, w, (2 * max_displacement + 1) ** 2] of costs for
    all displacements.
  """

  # Set maximum displacement and compute the number of image shifts.
  _, height, width, _ = features1.shape.as_list()
  if max_displacement <= 0 or max_displacement >= height:
    raise ValueError(f'Max displacement of {max_displacement} is too large.')

  max_disp = max_displacement
  num_shifts = 2 * max_disp + 1

  # Pad features2 and shift it while keeping features1 fixed to compute the
  # cost volume through correlation.

  # Pad features2 such that shifts do not go out of bounds.
  features2_padded = tf.pad(
      tensor=features2,
      paddings=[[0, 0], [max_disp, max_disp], [max_disp, max_disp], [0, 0]],
      mode='CONSTANT')
  cost_list = []
  for i in range(num_shifts):
    for j in range(num_shifts):
      corr = tf.reduce_mean(
          input_tensor=features1 *
          features2_padded[:, i:(height + i), j:(width + j), :],
          axis=-1,
          keepdims=True)
      cost_list.append(corr)
  cost_volume = tf.concat(cost_list, axis=-1)
  return cost_volume


def flow_to_warp(flow):
    height, width = flow.shape.as_list()[-3:-1]
    i_grid, j_grid = tf.meshgrid(
        tf.linspace(0.0, height - 1.0, int(height)),
        tf.linspace(0.0, width - 1.0, int(width)),
        indexing='ij')
    grid = tf.stack([i_grid, j_grid], axis=2)

    warp = grid + flow
    # warp=[b,h,w,2]

    return warp


def apply_warps_stop_grad(sources, warps, level):
    """Apply all warps on the correct sources."""

    warped = resample(tf.stop_gradient(sources), warps)

    return warped

def mask_invalid(coords):
  """Mask coordinates outside of the image.

  Valid = 1, invalid = 0.

  Args:
    coords: a 4D float tensor of image coordinates.

  Returns:
    The mask showing which coordinates are valid.
  """
  coords_rank = len(coords.shape)
  if coords_rank != 4:
    raise NotImplementedError()
  max_height = float(coords.shape[-3] - 1)
  max_width = float(coords.shape[-2] - 1)
  mask = tf.logical_and(
      tf.logical_and(coords[:, :, :, 0] >= 0.0,
                     coords[:, :, :, 0] <= max_height),
      tf.logical_and(coords[:, :, :, 1] >= 0.0,
                     coords[:, :, :, 1] <= max_width))
  mask = tf.cast(mask, dtype=tf.float32)[:, :, :, None]
  return mask


def resample(source, coords):
    """Resample the source image at the passed coordinates.

    Args:
      source: tf.tensor, batch of images to be resampled.
      coords: tf.tensor, batch of coordinates in the image.

    Returns:
      The resampled image.
    """
    output = resampler(source, coords[:, :, :, ::-1])

    return output


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
    with tf.name_scope(name + '/unstack_warp'):  # 返回在定义 Python 操作时使用的上下文管理器
        warp_x, warp_y = tf.unstack(warp, axis=-1)  # 将秩为 R 的张量的给定维度出栈为秩为 (R-1) 的张量.
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
    params_shape = tf.shape(params)  # 形状
    indices_shape = tf.shape(indices)
    slice_dimensions = indices_shape[-1]

    max_index = params_shape[:slice_dimensions] - 1
    min_index = tf.zeros_like(max_index, dtype=tf.int32)  # 返回一个全部为0的张量，大小和输入的一样

    clipped_indices = tf.clip_by_value(indices, min_index, max_index)  # 可以将一个张量中的数值限制在一个范围之内

    # Check whether each component of each index is in range [min, max], and
    # allow an index only if all components are in range:
    mask = tf.reduce_all(  # 计算一个张量在维度上元素的“逻辑和”
        tf.logical_and(indices >= min_index, indices <= max_index), -1)
    mask = tf.expand_dims(mask, -1)
    # 对dtype 执行一个安全的饱和cast的value.
    return (tf.cast(mask, dtype=params.dtype) *
            tf.gather_nd(params, clipped_indices))


# 将参数中的切片收集到由索引指定的形状的张量中.

class Dis_model(Model):
    def __init__(self,
                 height=FLAGS.height,
                 width=FLAGS.width,
                 batch_size=FLAGS.batch_size,
                 checkpoint_dir=FLAGS.checkpoint_dir_dis,
                 learning_rate=0.0002,):
        super(Dis_model, self).__init__()
        self._height = height
        self._width = width
        self._batch_size = batch_size
        self._dtype_policy = tf.keras.mixed_precision.experimental.Policy(
            'float32')
        self._leaky_relu_alpha = 0.1
        self._normalize_before_cost_volume = True
        # learning rate setting
        self._optimizer_type = 'adam'
        self._learning_rate = learning_rate
        self._make_or_reset_optimizer()

        # Set up checkpointing.
        self._make_or_reset_checkpoint()
        self.update_checkpoint_dir(checkpoint_dir)

        self._feature_model = uflow_model1.PWCFeaturePyramid(
            level1_num_layers=3,
            level1_num_filters=16,
            level1_num_1x1=0,
            original_layer_sizes=False,
            num_levels=5,
            channel_multiplier=1.,
            pyramid_resolution='half',
            use_bfloat16=False)

        self.convs_1 = []

        # for c, d in [(32, 3), (64, 3), (128, 3), (64, 3)]:
        for c, d in [(32, 3), (64, 3), (128, 3)]:
        # for c, d in [(32, 3), (64, 3)]:
        # for c, d in [(32, 3)]:
            self.convs_1.append(
                Conv2D(
                    int(c * FLAGS.channel_multiplier),
                    kernel_size=(3, 3),
                    strides=1,
                    padding='same',
                    dtype=self._dtype_policy
                )
            )
            self.convs_1.append(BatchNormalization())
            self.convs_1.append(LeakyReLU(
                    alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))

        self.convs_2 = []
        # self.convs_2.append(Conv2D(
        #     32, kernel_size=(3, 3), strides=1,
        #     padding='same', dtype=self._dtype_policy))
        # self.convs_2.append(BatchNormalization())
        # self.convs_2.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_2.append(Conv2D(
        #     64, kernel_size=(3, 3), strides=1,
        #     padding='same', dtype=self._dtype_policy))
        # self.convs_2.append(BatchNormalization())
        # self.convs_2.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_2.append(Conv2D(
        #     128, kernel_size=(3, 3), strides=1,
        #     padding='same', dtype=self._dtype_policy))
        # self.convs_2.append(BatchNormalization())
        # self.convs_2.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))

        self.convs_2.append(Conv2D(
            64, kernel_size=(3, 3), strides=1,
            padding='same', dtype=self._dtype_policy))
        self.convs_2.append(BatchNormalization())
        self.convs_2.append(LeakyReLU(
            alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))

        self.convs_2.append(Conv2D(
            32, kernel_size=(3, 3), strides=1,
            padding='same', dtype=self._dtype_policy))
        self.convs_2.append(BatchNormalization())
        self.convs_2.append(LeakyReLU(
            alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))

        self.convs_2.append(tf.keras.layers.Dropout(0.5))

        self.convs_2.append(Conv2D(
            1, kernel_size=(1, 1), strides=1,
            padding='same', dtype=self._dtype_policy))
        self.convs_2.append(BatchNormalization())
        self.convs_2.append(LeakyReLU(
            alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))
        # self.convs_2.append(Conv2D(
        #     1, kernel_size=(1, 1), strides=1,
        #     padding='valid', dtype=self._dtype_policy))
        # self.convs_2.append(BatchNormalization())
        # self.convs_2.append(LeakyReLU(
        #     alpha=self._leaky_relu_alpha, dtype=self._dtype_policy))

        self.convs_3 = Conv1D(1, 1, activation=None)

    def save(self):
        """Saves a model checkpoint."""
        self._manager.save()

    def _make_or_reset_optimizer(self):
        if self._optimizer_type == 'adam':
            self._optimizer_origin = tf.compat.v1.train.AdamOptimizer(
                self._learning_rate, name='Optimizer')
            self._optimizer = tf.compat.v1.train.AdamOptimizer(
                self._learning_rate, name='Optimizer')

        elif self._optimizer_type == 'sgd':
            self._optimizer_origin = tf.compat.v1.train.GradientDescentOptimizer(
                self._learning_rate, name='Optimizer')
            self._optimizer = tf.compat.v1.train.AdamOptimizer(
                self._learning_rate, name='Optimizer')
        else:
            raise ValueError('Optimizer "{}" not yet implemented.'.format(
                self._optimizer_type))

    def update_checkpoint_dir(self, checkpoint_dir):
        """Changes the checkpoint directory for saving and restoring."""
        self._manager = tf.train.CheckpointManager(
            self._checkpoint, directory=checkpoint_dir, max_to_keep=1)

    def restore(self, reset_optimizer=False, reset_global_step=False):
        """Restores a saved model from a checkpoint."""
        status = self._checkpoint.restore(self._manager.latest_checkpoint)
        try:
            status.assert_existing_objects_matched()
        except AssertionError as e:
            print('Error while attempting to restore PF models:', e)
        if reset_optimizer:
            self._make_or_reset_optimizer()
            self._make_or_reset_checkpoint()
        if reset_global_step:
            tf.compat.v1.train.get_or_create_global_step().assign(0)

    def _make_or_reset_checkpoint(self):
        self._checkpoint = tf.train.Checkpoint(
            optimizer=self._optimizer,
            model=self,
            optimizer_step=tf.compat.v1.train.get_or_create_global_step())

    def _Feature_contract_layers(self, h, w):#image BHW3
        """build the moudle for contract feature from image"""
        layers = []
        for c, d in [(32, 3), (64, 3), (128, 3), (256, 3)]:
            layers.append(
                self.conv2d(
                    int(c * FLAGS.channel_multiplier),
                    kernel_size=(3, 3),
                    strides=2,
                    padding='valid',
                    dtype=self._dtype_policy
                )
            )
            layers.append(self.leaky_relu)

        return layers

    def call(self, image_1, image_3):#B*H*W*3 *3
    #def call(self, image):

        # B, H, W, _ = image_1.shape

        # 特征提取
        # feature_layers = self._Feature_contract_layers(self._height, self._width)
        # image_1 = image[:, :, :, :, 0]
        # image_2 = image[:, :, :, :, 1]
        # image_3 = image[:, :, :, :, 2]
        # feature_1 = tf.concat([image_1, image_3], axis=-1)
        # feature_2 = tf.concat([image_3, image_1], axis=-1)

        # feature_1 = self._feature_model(image_1)
        # feature_2 = self._feature_model(image_3)
        feature_1 = image_1
        feature_2 = image_3
        # print('1:', image_1) # 1 * 384 * 512 * 3

        for layer in self.convs_1:
            feature_1 = layer(feature_1)
        # print('2: ', feature_1)# 1 * 376 * 504 *256

        for layer in self.convs_1:
            feature_2 = layer(feature_2)

        # 正向为1
        # feature11_normalized, feature31_normalized  = normalize_features([feature_1, feature_2],
        #                            normalize=self._normalize_before_cost_volume,
        #                            center=self._normalize_before_cost_volume,
        #                            moments_across_channels=True,
        #                            moments_across_images=True)

        feature1 = tf.concat([feature_1, feature_2], axis=-1)
        # print('3: ', feature1)# 1 * 376 * 504 * 512

        # 卷积提取信息
        out_put1 = feature1

        for layer in self.convs_2:
            out_put1 = layer(out_put1)
        # print('4: ', out_put1)# 1 * 183* 247 * 512

        # out_put1 = tf.reshape(out_put1, (self._batch_size, 1, -1))
        #
        # # compute the possible values
        # out_put1 = self.convs_3(out_put1)
        #
        out_put_logit1 = out_put1

        out_put1 = tf.nn.sigmoid(out_put_logit1)

        # 反向为0
        # feature32_normalized, feature12_normalized = normalize_features([feature_2, feature_1],
        #                                                                 normalize=self._normalize_before_cost_volume,
        #                                                                 center=self._normalize_before_cost_volume,
        #                                                                 moments_across_channels=True,
        #                                                                 moments_across_images=True)

        feature2 = tf.concat([feature_2, feature_1], axis=-1)

        # 卷积提取信息
        out_put2 = feature2

        for layer in self.convs_2:
            out_put2 = layer(out_put2)

        # out_put2 = tf.reshape(out_put2, (self._batch_size, 1, -1))

        # compute the possible values
        #out_put2 = self.convs_3(out_put2)

        out_put_logit2 = out_put2

        out_put2 = tf.nn.sigmoid(out_put_logit2)



        return out_put1, out_put2, out_put_logit1, out_put_logit2

    def compute_loss(self, image_1, image_2, flow1, occlusion1, flow2):
        a = image_1.get_shape()
        B, H, W, C = a.as_list()

        warp1 = flow_to_warp(flow1)
        mask1 = mask_invalid(warp1)

        warp = flow_to_warp(flow1 + flow2)
        image_3 = apply_warps_stop_grad(image_1, warp, 'resampler')

        occlusion1 = tf.stop_gradient((1 - occlusion1) * mask1)

        occlusion1 = tf.where(tf.less(occlusion1, 0.9), tf.zeros_like(occlusion1), tf.ones_like(occlusion1))
        occlusion1 = tf.squeeze(occlusion1, axis=-1)

        image_11 = image_1[:, :, :, 0]
        image_11 = tf.where(tf.equal(occlusion1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25),
                            image_11)
        image_12 = image_1[:, :, :, 1]
        image_12 = tf.where(tf.equal(occlusion1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25),
                            image_12)
        image_13 = image_1[:, :, :, 2]
        image_13 = tf.where(tf.equal(occlusion1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25),
                            image_13)

        image_1 = tf.stack([image_11, image_12, image_13], axis=3)

        image_31 = image_3[:, :, :, 0]
        image_31 = tf.where(tf.equal(occlusion1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25),
                            image_31)
        image_32 = image_3[:, :, :, 1]
        image_32 = tf.where(tf.equal(occlusion1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25),
                            image_32)
        image_33 = image_3[:, :, :, 2]
        image_33 = tf.where(tf.equal(occlusion1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25),
                            image_33)

        image_3 = tf.stack([image_31, image_32, image_33], axis=3)

        out_put_true, out_put_false, out_put_true_logit, out_put_false_logit = self.call(image_1, image_3)

        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_true_logit,
                                                    labels=tf.ones_like(out_put_true_logit))) + \
                 tf.reduce_mean(
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_false_logit,
                                                             labels=tf.zeros_like(out_put_false_logit)))
        d_loss = 1 * d_loss

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_false_logit,
                                                    labels=tf.ones_like(out_put_false_logit)))

        return d_loss, g_loss

    def train_step(self, images, flow1, occlusion1, flow2, occlusion2):
        image_1 = images[:, 0]# 1*384*512*3
        image_2 = images[:, 1]

        with tf.GradientTape() as tape:
            d_loss1, g_loss1 = self.compute_loss(image_1, image_2, flow1, occlusion1, flow2)
            d_loss2, g_loss2 = self.compute_loss(image_2, image_1, flow2, occlusion2, flow1)

            d_loss = (d_loss1 + d_loss2) / 2.
            g_loss = (g_loss1 + g_loss2) / 2.

            variables = self.trainable_variables

            grads = tape.gradient(d_loss, variables)

            self._optimizer.apply_gradients(zip(grads, variables))

        return d_loss, g_loss

    def train_step1(self, images, flow1, occlusion1):
        #B, _, H, W, _ = out_put_true.shape

        image_1 = images[:, 0]# 1*384*512*3
        image_2 = images[:, 1]

        warp1 = flow_to_warp(flow1)
        mask1 = mask_invalid(warp1)
        image_3 = apply_warps_stop_grad(image_2, warp1, 'resampler')

        a = image_1.get_shape()
        B, H, W, C = a.as_list()

        occlusion1 = tf.stop_gradient((1 - occlusion1) * mask1)

        occlusion1 = tf.where(tf.less(occlusion1, 0.9), tf.zeros_like(occlusion1), tf.ones_like(occlusion1))

        image_11 = image_1[:, :, :, 0] * tf.squeeze(occlusion1, axis=-1)
        image_11 = tf.where(tf.equal(image_11, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_11)
        image_12 = image_1[:, :, :, 1] * tf.squeeze(occlusion1, axis=-1)
        image_12 = tf.where(tf.equal(image_12, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_12)
        image_13 = image_1[:, :, :, 2] * tf.squeeze(occlusion1, axis=-1)
        image_13 = tf.where(tf.equal(image_13, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_13)

        image_1 = tf.stack([image_11, image_12, image_13], axis=3)

        image_31 = image_3[:, :, :, 0] * tf.squeeze(occlusion1, axis=-1)
        image_31 = tf.where(tf.equal(image_31, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_31)
        image_32 = image_3[:, :, :, 1] * tf.squeeze(occlusion1, axis=-1)
        image_32 = tf.where(tf.equal(image_32, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_32)
        image_33 = image_3[:, :, :, 2] * tf.squeeze(occlusion1, axis=-1)
        image_33 = tf.where(tf.equal(image_33, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_33)

        image_3 = tf.stack([image_31, image_32, image_33], axis=3)


        with tf.GradientTape() as tape:
            out_put_true, out_put_false, out_put_true_logit, out_put_false_logit = self.call(image_1, image_3)
            d_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_true_logit, labels=tf.ones_like(out_put_true_logit))) + \
                     tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_false_logit, labels=tf.zeros_like(out_put_false_logit)))
            d_loss = 1 * d_loss

            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_false_logit, labels=tf.ones_like(out_put_false_logit)))

            # d_loss = -tf.reduce_mean(tf.math.log(output_true)) - tf.reduce_mean(tf.math.log(1 - output_false))

            variables = self.trainable_variables

            grads = tape.gradient(d_loss, variables)

            self._optimizer.apply_gradients(zip(grads, variables))

        return d_loss, out_put_true, out_put_false, g_loss

    def test(self, images, flow):
        #B, _, H, W, _ = out_put_true.shape

        image_1 = images[:, 0]
        image_2 = images[:, 1]

        warp = flow_to_warp(flow)
        image_3 = apply_warps_stop_grad(image_2, warp, 'resampler')

        #output_true, output_false = self.call(image_1, image_2, image_3)

        out_put_true, out_put_false, out_put_true_logit, out_put_false_logit = self.call(image_1, image_2, image_3)
        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_true_logit, labels=tf.ones_like(out_put_true_logit))) + \
                 tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_false_logit, labels=tf.zeros_like(out_put_false_logit)))
        # d_loss = 0.01 * d_loss

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_false_logit, labels=tf.ones_like(out_put_false_logit)))

        # d_loss = -tf.reduce_mean(tf.math.log(output_true)) - tf.reduce_mean(tf.math.log(1 - output_false))


        return d_loss, out_put_true, out_put_false, g_loss

    def g_loss(self, image_1, image_2, flow, mask1):


        a = image_1.get_shape()
        B, H, W, C = a.as_list()

        warp = flow_to_warp(flow)

        image_3 = apply_warps_stop_grad(image_1, warp, 'resampler')

        mask1 = tf.where(tf.less(mask1, 0.8), tf.zeros_like(mask1), tf.ones_like(mask1))
        mask1 = tf.squeeze(mask1, axis=-1)

        image_11 = image_1[:, :, :, 0]
        image_11 = tf.where(tf.equal(mask1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_11)
        image_12 = image_1[:, :, :, 1]
        image_12 = tf.where(tf.equal(mask1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_12)
        image_13 = image_1[:, :, :, 2]
        image_13 = tf.where(tf.equal(mask1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_13)

        image_1 = tf.stack([image_11, image_12, image_13], axis=3)


        image_31 = image_3[:, :, :, 0]
        image_31 = tf.where(tf.equal(mask1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_31)
        image_32 = image_3[:, :, :, 1]
        image_32 = tf.where(tf.equal(mask1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_32)
        image_33 = image_3[:, :, :, 2]
        image_33 = tf.where(tf.equal(mask1, 0.0), tf.random.truncated_normal([B, H, W], mean=0.5, stddev=0.25), image_33)

        image_3 = tf.stack([image_31, image_32, image_33], axis=3)

        out_put_true, out_put_false, out_put_true_logit, out_put_false_logit = self.call(image_1, image_3)

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_put_false_logit,
                                                    labels=tf.ones_like(out_put_false_logit)))

        return g_loss


    def vision(self, images, flow1, flow2, occlusion1, occlusion2, plot_dir, index):

        image1 = images[:, 0]  # 1*384*512*3
        image2 = images[:, 1]

        warp1 = flow_to_warp(flow1)
        mask1 = mask_invalid(warp1)
        image3 = apply_warps_stop_grad(image2, warp1, 'resampler')


        occlusion1 = (1 - occlusion1) * mask1
        occlusion1 = tf.where(tf.less(occlusion1, 0.8), tf.zeros_like(occlusion1), tf.ones_like(occlusion2))

        warp2 = flow_to_warp(flow2)
        mask2 = mask_invalid(warp2)
        occlusion2 = (1 - occlusion2) * mask2
        occlusion2 = tf.where(tf.less(occlusion2, 0.8), tf.zeros_like(occlusion2), tf.ones_like(occlusion2))

        image_11 = image1[:, :, :, 0] * tf.squeeze(occlusion1, axis=-1)
        image_12 = image1[:, :, :, 1] * tf.squeeze(occlusion1, axis=-1)
        image_13 = image1[:, :, :, 2] * tf.squeeze(occlusion1, axis=-1)

        image_1 = tf.stack([image_11, image_12, image_13], axis=3)

        image_21 = image2[:, :, :, 0] * tf.squeeze(occlusion2, axis=-1)
        image_22 = image2[:, :, :, 1] * tf.squeeze(occlusion2, axis=-1)
        image_23 = image2[:, :, :, 2] * tf.squeeze(occlusion2, axis=-1)

        image_2 = tf.stack([image_21, image_22, image_23], axis=3)

        image_31 = image3[:, :, :, 0] * tf.squeeze(occlusion1, axis=-1)
        image_32 = image3[:, :, :, 1] * tf.squeeze(occlusion1, axis=-1)
        image_33 = image3[:, :, :, 2] * tf.squeeze(occlusion1, axis=-1)

        image_3 = tf.stack([image_31, image_32, image_33], axis=3)

        image1 = tf.squeeze(image1, axis=0).numpy()
        image2 = tf.squeeze(image2, axis=0).numpy()
        image3 = tf.squeeze(image3, axis=0).numpy()
        image_1 = tf.squeeze(image_1, axis=0).numpy()
        image_2 = tf.squeeze(image_2, axis=0).numpy()
        image_3 = tf.squeeze(image_3, axis=0).numpy()
        flow1 = tf.squeeze(flow1, axis=0).numpy()
        flow2 = tf.squeeze(flow2, axis=0).numpy()
        occlusion1 = tf.squeeze(occlusion1, axis=0).numpy()
        occlusion2 = tf.squeeze(occlusion2, axis=0).numpy()

        uflow_plotting.complete_vision(plot_dir,
                        index,
                        image1,
                        image2,
                        image3,
                        image_1,
                        image_2,
                        image_3,
                        flow1,
                        flow2,
                        occlusion1,
                        occlusion2)



def main(argv):
    image_1 = tf.fill([1, 384, 512, 3], 1.)
    image_2 = tf.fill([1, 384, 512, 3], 1.)
    image_3 = tf.fill([1, 384, 512, 3], 0.)
    image = tf.fill([1, 384, 512, 3, 3], 0.)

    dis = Dis_model()

    # out = dis._feature_model(image_1)
    #
    # print(out)

    out1, out2, _, _ = dis(image_1, image_3)
    # out1, out2, _, _ = dis.call(image_2)
    print(out1, out2)

    #
    # input_signature = [
    #     [tf.TensorSpec(shape=(1,) + (512, 224, 3))],
    #     [tf.TensorSpec(shape=(1,) + (512, 224, 3))],
    #     [tf.TensorSpec(shape=(1,) + (512, 224, 3))]
    # ]

    # print(dis.trainable_variables)
    # dis.build(input_shape=(1, 1024, 424, 3, 3))
    # dis.summary()

if __name__ == '__main__':
  app.run(main)