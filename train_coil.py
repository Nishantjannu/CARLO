import numpy as np
import tensorflow as tf
import argparse
from utils import *
from tensorflow.keras.initializers import he_uniform, zeros, glorot_uniform

tf.config.run_functions_eagerly(True)

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()

        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the CoIL network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT: You should use either of the following for weight initialization:
        #         - tf.keras.initializers.GlorotUniform (this is what we tried)
        #         - tf.keras.initializers.GlorotNormal
        #         - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal

        # n_nodes_hidden = 30
        act_fun = "tanh"
        kernel_init = glorot_uniform
        self.dense1 = tf.keras.layers.Dense(in_size, activation=act_fun, kernel_initializer=kernel_init, bias_initializer=zeros)
        self.dense2 = tf.keras.layers.Dense(20, activation=act_fun, kernel_initializer=kernel_init, bias_initializer=zeros)
        self.dense3 = tf.keras.layers.Dense(20, activation=act_fun, kernel_initializer=kernel_init, bias_initializer=zeros)

        # Thinking about using simply the last layer as different for the structure, might need larger subnets though
        self.dense_left_1 = tf.keras.layers.Dense(10, activation=act_fun, kernel_initializer=kernel_init, bias_initializer=zeros)
        self.dense_left_2 = tf.keras.layers.Dense(out_size, kernel_initializer=glorot_uniform, bias_initializer=zeros)
        self.dense_straight_1 = tf.keras.layers.Dense(10, activation=act_fun, kernel_initializer=kernel_init, bias_initializer=zeros)
        self.dense_straight_2 = tf.keras.layers.Dense(out_size, kernel_initializer=glorot_uniform, bias_initializer=zeros)
        self.dense_right_1 = tf.keras.layers.Dense(10, activation=act_fun, kernel_initializer=kernel_init, bias_initializer=zeros)
        self.dense_right_2 = tf.keras.layers.Dense(out_size, kernel_initializer=glorot_uniform, bias_initializer=zeros)



        ########## Your code ends here ##########

    def call(self, x, u):
        x = tf.cast(x, dtype=tf.float32)
        u = tf.cast(u, dtype=tf.int8)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for (x,u) where:
        # - x is a (?, |O|) tensor that keeps a batch of observations
        # - u is a (?, 1) tensor (a vector indeed) that keeps the high-level commands (goals) to denote which branch of the network to use 
        # FYI: For the intersection scenario, u=0 means the goal is to turn left, u=1 straight, and u=2 right. 
        # HINT 1: Looping over all data samples may not be the most computationally efficient way of doing branching
        # HINT 2: While implementing this, we found tf.math.equal and tf.cast useful. This is not necessarily a requirement though.
        out1 = self.dense1(x)
        out2 = self.dense2(out1)
        out3 = self.dense3(out2)

        try:
            shape = [u.shape[0], u.shape[1]]
        except:
            shape = [1, 1]  # if u is a scalar
        zero_tens = tf.constant([0]*shape[0], dtype=tf.int8, shape=shape)
        mask_left = tf.math.equal(u, zero_tens)
        mask_left = tf.cast(mask_left, tf.float32)
        mask_straight = tf.cast(tf.math.equal(u, tf.constant([1]*shape[0], shape=shape, dtype=tf.int8)), tf.float32)
        mask_right = tf.cast(tf.math.equal(u, tf.constant([2]*shape[0], shape=shape, dtype=tf.int8)), tf.float32)

        out_left_1 = self.dense_left_1(out3)
        out_left = self.dense_left_2(out_left_1)
        out_straight_1 = self.dense_straight_1(out3)
        out_straight = self.dense_straight_2(out_straight_1)
        out_right_1 = self.dense_right_1(out3)
        out_right = self.dense_right_2(out_right_1)

        out = tf.math.multiply(mask_left, out_left) \
              + tf.math.multiply(mask_straight, out_straight) \
              + tf.math.multiply(mask_right, out_right)

        return out

        ########## Your code ends here ##########


def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations & goals,
    # - y is the actions the expert took for the corresponding batch of observations & goals
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally

    # Transparent way
    N = y.shape[0]
    weights = tf.constant([15., 1.])  # First element here corresponds to the steering
    total_elementwise_loss = tf.math.reduce_sum(tf.math.square(y_est - y), axis=0)
    loss_val = (1/N) * tf.tensordot(weights, total_elementwise_loss, axes=1)

    return loss_val

    ########## Your code ends here ##########


def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]

    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y, u):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model (note both x and u are inputs now)
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients

        # src: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
        with tf.GradientTape() as g:
            g.watch(nn_model.variables)

            # 1.
            y_est = nn_model.call(x, u)

            # 2.
            current_loss = loss(y_est, y)

            # 3.
            gradient = g.gradient(current_loss, nn_model.variables)

            # 4.
            optimizer.apply_gradients(zip(gradient, nn_model.variables))

        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y, u in train_data:
            train_step(x, y, u)

    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'], data['u_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    args.goal = 'all'

    maybe_makedirs("./policies")

    data = load_data(args)

    nn(data, args)
