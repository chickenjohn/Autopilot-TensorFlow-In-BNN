import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data
from ConvModel import ConvModel
import argparse

LOGDIR = './save'
LEARNING_RATE = 1e-4

def get_arguments():

  parser = argparse.ArgumentParser(description='ConvNet training')
  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                      help='Learning rate for training.')
  parser.add_argument('--drop_out', type=bool, default=False,
                      help='Dropout switcher.')
  return parser.parse_args()


def main():

  args = get_arguments()

  sess = tf.InteractiveSession()
  model = ConvModel(drop_out=True)
  L2NormConst = 0.001

  train_vars = tf.trainable_variables()

  loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))) + \
      tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
  train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
  sess.run(tf.global_variables_initializer())

  # create a summary to monitor cost tensor
  tf.summary.scalar("loss", loss)
  # merge all summaries into a single op
  merged_summary_op = tf.summary.merge_all()

  saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)

  # op to write logs to Tensorboard
  logs_path = './logs'
  summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

  epochs = 30
  batch_size = 100

  # train over the dataset about 30 times
  for epoch in range(epochs):
      for i in range(int(driving_data.num_images / batch_size)):
          xs, ys = driving_data.LoadTrainBatch(batch_size)
          train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.3})
          if i % 10 == 0:
              xs, ys = driving_data.LoadValBatch(batch_size)
              loss_value = loss.eval(feed_dict={model.x: xs, model.y_: ys,  model.keep_prob: 1.0})
              print("Epoch: %d, Step: %d, Loss: %g" %
                    (epoch, epoch * batch_size + i, loss_value))

          # write logs at every iteration
          summary = merged_summary_op.eval(feed_dict={model.x: xs, model.y_: ys,  model.keep_prob: 1.0})
          summary_writer.add_summary(
              summary, epoch * driving_data.num_images / batch_size + i)

          if i % batch_size == 0:
              if not os.path.exists(LOGDIR):
                  os.makedirs(LOGDIR)
              checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
              filename = saver.save(sess, checkpoint_path)
      print("Model saved in file: %s" % filename)

  print("Run the command line:\n"
        "--> tensorboard --logdir=./logs "
        "\nThen open http://0.0.0.0:6006/ into your web browser")

if __name__ == '__main__':
    main()
