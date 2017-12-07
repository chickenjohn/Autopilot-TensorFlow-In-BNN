import tensorflow as tf
import scipy.misc
from ConvModel import ConvModel
from subprocess import call
import math
from numpy import genfromtxt

sess = tf.InteractiveSession()
model = ConvModel(drop_out=True, relu=False, is_training=False)
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

smoothed_angle = 0

filename = 'dataset/data.txt'

data=genfromtxt(filename, delimiter=' ', names=['file', 'angle'])

intervention = 0.0
degree_delta = 0.0

i = 0
while i < 45406:
    print("index: %d\n" % i)
    full_image = scipy.misc.imread("dataset/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    degree_delta = math.fabs(degrees - float(data['angle'][i]) * 100 / scipy.pi)
    if degree_delta>10.0 :
        intervention += 1.0
    print("Predicted steering angle: " + str(degrees) + " degrees")
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    i += 1

print("number of intervention: " + str(intervention) + " and autonomy: " + str(1-intervention/45406))
