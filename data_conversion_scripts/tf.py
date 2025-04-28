import tensorflow as tf

def add_layer(inputs,in_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size])+0.1)
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matumul(inputs,Weights)+biases
    if actionvation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs