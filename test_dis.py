from absl import app
from absl import flags

import tensorflow as tf
import sys
import gin
import time

from uflow import uflow_data
from uflow import uflow_main
from uflow import dis_net
from uflow import uflow_plotting
from uflow import uflow_flags

FLAGS = flags.FLAGS

def create_discriminator():
    dis = dis_net.Dis_model(FLAGS.checkpoint_dir_dis)
    return dis

def main(unused_argv):
    kitti_EPE_1 = 100.0  # kitti-2015
    sintel_EPE_1 = 100.0  # sintel-clean
    sintel_EPE_2 = 100.0  # sintel-final
    chairs_EPE = 100.0
    num = 5

    if FLAGS.no_tf_function:
        tf.config.experimental_run_functions_eagerly(True)
        print('TFFUNCTION DISABLED')

    gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
    # Make directories if they do not exist yet.
    if FLAGS.checkpoint_dir and not tf.io.gfile.exists(FLAGS.checkpoint_dir):
        print('Making new checkpoint directory', FLAGS.checkpoint_dir)
        tf.io.gfile.makedirs(FLAGS.checkpoint_dir)
    if FLAGS.checkpoint_dir1 and not tf.io.gfile.exists(FLAGS.checkpoint_dir1):
        print('Making new checkpoint directory', FLAGS.checkpoint_dir1)
        tf.io.gfile.makedirs(FLAGS.checkpoint_dir1)
    if FLAGS.checkpoint_dir2 and not tf.io.gfile.exists(FLAGS.checkpoint_dir2):
        print('Making new checkpoint directory', FLAGS.checkpoint_dir2)
        tf.io.gfile.makedirs(FLAGS.checkpoint_dir2)
    if FLAGS.checkpoint_dir_dis and not tf.io.gfile.exists(FLAGS.checkpoint_dir_dis):
        print('Making new checkpoint directory', FLAGS.checkpoint_dir_dis)
        tf.io.gfile.makedirs(FLAGS.checkpoint_dir_dis)
    if FLAGS.plot_dir and not tf.io.gfile.exists(FLAGS.plot_dir):
        print('Making new plot directory', FLAGS.plot_dir)
        tf.io.gfile.makedirs(FLAGS.plot_dir)

    uflow = uflow_main.create_uflow()
    dis = create_discriminator()

    if not FLAGS.from_scratch:
        # First restore from init_checkpoint_dir, which is only restored from but
        # not saved to, and then restore from checkpoint_dir if there is already
        # a model there (e.g. if the run was stopped and restarted).
        if FLAGS.init_checkpoint_dir:
            print('Initializing model from checkpoint {}.'.format(
                FLAGS.init_checkpoint_dir))
            uflow.update_checkpoint_dir(FLAGS.init_checkpoint_dir)
            uflow.restore(
                reset_optimizer=FLAGS.reset_optimizer,
                reset_global_step=FLAGS.reset_global_step)
            uflow.update_checkpoint_dir(FLAGS.checkpoint_dir)

        elif FLAGS.checkpoint_dir:
            print('Restoring model from checkpoint {}.'.format(FLAGS.checkpoint_dir))
            uflow.restore()
    else:
        print('Starting from scratch.')

    if FLAGS.updata_dis:
        print('Restoring discriminator model from checkpoint_dis {}.'.format(FLAGS.checkpoint_dir_dis))
        dis.restore(
            reset_optimizer=FLAGS.reset_optimizer,
            reset_global_step=FLAGS.reset_global_step
        )


    if FLAGS.eval_on:
        print('Making eval datasets and eval functions.')
        evaluate, _ = uflow_data.make_eval_function(
            FLAGS.eval_on,
            FLAGS.height,
            FLAGS.width,
            progress_bar=True,
            plot_dir=FLAGS.plot_dir,
            num_plots=50)

    if FLAGS.train_on:
        # Build training iterator.
        print('Making training iterator.')
        train_it = uflow_data.make_train_iterator(
            FLAGS.train_on,
            FLAGS.height,
            FLAGS.width,
            FLAGS.shuffle_buffer_size,
            FLAGS.batch_size,
            FLAGS.seq_len,
            crop_instead_of_resize=FLAGS.crop_instead_of_resize,
            apply_augmentation=True,
            include_ground_truth=FLAGS.use_supervision,
            resize_gt_flow=FLAGS.resize_gt_flow_supervision,
            include_occlusions=FLAGS.use_gt_occlusions
        )

        if FLAGS.use_supervision:
            weights = {'supervision': 1.}
        else:
            weights = {
                'photo': FLAGS.weight_photo,
                'ssim': FLAGS.weight_ssim,
                'census': FLAGS.weight_census,
                'smooth1': FLAGS.weight_smooth1,
                'smooth2': FLAGS.weight_smooth2,
                'edge_constant': FLAGS.smoothness_edge_constant,
            }
            weights = {
                k: v for (k, v) in weights.items() if v > 1e-7 or k == 'edge_constant'
            }

        def weight_selfsup_fn():
            step = tf.compat.v1.train.get_or_create_global_step(
            ) % FLAGS.selfsup_step_cycle
            # Start self-supervision only after a certain number of steps.
            # Linearly increase self-supervision weight for a number of steps.
            ramp_up_factor = tf.clip_by_value(
                float(step - (FLAGS.selfsup_after_num_steps - 1)) /
                float(max(FLAGS.selfsup_ramp_up_steps, 1)), 0., 1.)
            return FLAGS.weight_selfsup * ramp_up_factor

        distance_metrics = {
            'photo': FLAGS.distance_photo,
            'census': FLAGS.distance_census,
        }

        print('Starting training loop.')
        log = dict()
        epoch = 0

        teacher_feature_model = None
        teacher_flow_model = None
        test_frozen_flow = None

        while True:
            current_step = tf.compat.v1.train.get_or_create_global_step().numpy()
            occ_active = {
                'uflow':
                    FLAGS.occlusion_estimation == 'uflow',
                'brox':
                    current_step > FLAGS.occ_after_num_steps_brox,
                'wang':
                    current_step > FLAGS.occ_after_num_steps_wang,
                'wang4':
                    current_step > FLAGS.occ_after_num_steps_wang,
                'wangthres':
                    current_step > FLAGS.occ_after_num_steps_wang,
                'wang4thres':
                    current_step > FLAGS.occ_after_num_steps_wang,
                'fb_abs':
                    current_step > FLAGS.occ_after_num_steps_fb_abs,
                'forward_collision':
                    current_step > FLAGS.occ_after_num_steps_forward_collision,
                'backward_zero':
                    current_step > FLAGS.occ_after_num_steps_backward_zero,
            }
            current_weights = {k: v for k, v in weights.items()}

            # Prepare self-supervision if it will be used in the next epoch.
            if FLAGS.weight_selfsup > 1e-7 and (
                    current_step % FLAGS.selfsup_step_cycle
            ) + FLAGS.epoch_length > FLAGS.selfsup_after_num_steps:

                # Add selfsup weight with a ramp-up schedule. This will cause a
                # recompilation of the training graph defined in uflow.train(...).
                current_weights['selfsup'] = weight_selfsup_fn

                # Freeze model for teacher distillation.
                if teacher_feature_model is None and FLAGS.frozen_teacher:
                    # Create a copy of the existing models and freeze them as a teacher.
                    # Tell uflow about the new, frozen teacher model.
                    teacher_feature_model, teacher_flow_model = uflow_main.create_frozen_teacher_models(
                        uflow)
                    uflow.set_teacher_models(
                        teacher_feature_model=teacher_feature_model,
                        teacher_flow_model=teacher_flow_model)
                    test_frozen_flow = uflow_main.check_model_frozen(
                        teacher_feature_model, teacher_flow_model, prev_flow_output=None)

                    # Check that the model actually is frozen.
                    if FLAGS.frozen_teacher and test_frozen_flow is not None:
                        uflow_main.check_model_frozen(
                            teacher_feature_model,
                            teacher_flow_model,
                            prev_flow_output=test_frozen_flow)

            #train procedure
            num_steps = FLAGS.epoch_length
            log1 = dict()

            start_time_data = time.time()

            for _, batch in zip(range(num_steps), train_it):
                stop_time_data = time.time()
                sys.stdout.write('.')
                sys.stdout.flush()

                images, labels = batch

                flow, occlusion = uflow.batch_infer_no_tf_function(images, infer_occlusion=True)

                # start_time_train_step = time.time()

                d_loss, out_put_true, out_put_false, g_loss = dis.train_step(images, flow, occlusion)

                print('d_loss: ', d_loss.numpy(), 'out_put_true: ', out_put_true.numpy(), 'out_put_false: ', out_put_false.numpy(), 'g_loss: ', g_loss.numpy())

                #g_loss = uflow.train_step(images, d_model_fake, d_logits_fake, weights=current_weights, distance_metrics=distance_metrics, occ_active=occ_active)

                # stop_time_train_step = time.time()
                #
                # losses = dict()
                # losses['d-loss'] = d_loss
                #
                # log_update = losses
                #
                # log_update['data-time'] = (stop_time_data - start_time_data) * 1000
                # log_update['train-time'] = (stop_time_train_step - start_time_train_step) * 1000

            #     for key in log_update:
            #         if key in log1:
            #             log1[key].append(log_update[key])
            #         else:
            #             log1[key] = [log_update[key]]
            #
                start_time_data = time.time()
            #
            # for key in log1:
            #     log1[key] = tf.reduce_mean(input_tensor=log1[key])
            #
            # sys.stdout.write('\n')
            # sys.stdout.flush()
            #
            # for key in log1:
            #     if key in log:
            #         log[key].append(log1[key])
            #     else:
            #         log[key] = [log1[key]]

            if FLAGS.checkpoint_dir and not FLAGS.no_checkpointing:
                uflow.save()

            if FLAGS.checkpoint_dir_dis and not FLAGS.no_checkpointing:
                dis.save()

            #uflow_plotting.print_log(log, epoch)

            # if FLAGS.eval_on and FLAGS.evaluate_during_train and epoch % 1 == 0:
            #     # Evaluate
            #     eval_results = evaluate(uflow)
            #     uflow_plotting.print_eval(eval_results)
            #     status = ''.join(
            #         ['{}: {:.6f}, '.format(key, eval_results[key]) for key in sorted(eval_results)])
            #     eval_on = FLAGS.eval_on
            #     for format_and_path in eval_on.split(';'):
            #         data_format, path = format_and_path.split(':')
            #
            #     if 'kitti' in data_format:
            #         EPE_1 = float(status.split(',')[1].split(':')[1][1:])
            #         if EPE_1 < kitti_EPE_1:  # kitti-2015
            #             uflow.save_1()
            #             kitti_EPE_1 = EPE_1
            #     elif 'sintel' in data_format:
            #         EPE_1 = float(status.split(',')[0].split(':')[1][1:])
            #         #EPE_2 = float(status.split(',')[6].split(':')[1][1:])
            #         if EPE_1 < sintel_EPE_1:  # sintel-clean
            #             uflow.save_1()
            #             sintel_EPE_1 = EPE_1
            #         # if EPE_2 < sintel_EPE_2:  # sintel-fianl
            #         #     uflow.save_2()
            #         #     sintel_EPE_2 = EPE_2
            #     elif 'chairs' in data_format:
            #         EPE = float(status[12:20])
            #         if EPE < chairs_EPE:
            #             uflow.save_1()
            #             chairs_EPE = EPE

            if current_step >= FLAGS.num_train_steps:
                break

            epoch += 1


if __name__ == '__main__':
  app.run(main)
