import tensorflow as tf
from sonnet.python.modules import relational_memory
import math


class RMeN(object):

    def __init__(self, vocab_size, embedding_size, batch_size, initialization, mem_slots, num_heads,
                 use_pos, attention_mlp_layers, head_size, num_filters):
        # Placeholders for input, output
        self.input_x = tf.compat.v1.placeholder(tf.int32, [batch_size, 3], name="input_h")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [batch_size, 1], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("embedding"):
            if initialization != []:
                self.input_feature = tf.compat.v1.get_variable(name="input_feature_1", initializer=initialization)
            else:
                self.input_feature = tf.compat.v1.get_variable(name="input_feature_2", shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=1234))

        # Embedding lookup
        self.emb = tf.nn.embedding_lookup(self.input_feature, self.input_x)

        if use_pos == 1:
            self.emb = add_positional_embedding(self.emb, 3, embedding_size)

        self.h_emb, self.r_emb, self.t_emb = tf.compat.v1.split(self.emb, num_or_size_splits=3, axis=1)

        self.h_emb = tf.squeeze(self.h_emb)
        self.r_emb = tf.squeeze(self.r_emb)
        self.t_emb = tf.squeeze(self.t_emb)
        #Relational Memory
        gen_mem = relational_memory.RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads,
                                                     gate_style='memory', attention_mlp_layers=attention_mlp_layers)

        init_states = gen_mem.initial_state(batch_size=batch_size)

        mem_output1, memory_input_next_step = gen_mem(self.h_emb, init_states)
        mem_output2, memory_input_next_step = gen_mem(self.r_emb, memory_input_next_step)
        mem_output3, memory_input_next_step = gen_mem(self.t_emb, memory_input_next_step)

        mem_output1 = tf.compat.v1.reshape(mem_output1, [-1, 1, mem_output1.get_shape()[-1]])
        mem_output2 = tf.compat.v1.reshape(mem_output2, [-1, 1, mem_output2.get_shape()[-1]])
        mem_output3 = tf.compat.v1.reshape(mem_output3, [-1, 1, mem_output3.get_shape()[-1]])

        mem_output = tf.compat.v1.concat([mem_output1, mem_output2, mem_output3], axis=1)
        self.input_cnn = tf.expand_dims(mem_output, -1)

        #CNN decoder
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        with tf.name_scope("conv-maxpool"):
            W = tf.compat.v1.get_variable("W_conv", shape=[3, 1, 1, num_filters], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            b = tf.Variable(tf.zeros([num_filters]))
            conv = tf.nn.conv2d(self.input_cnn, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # Apply nonlinearity
            self.h_pool = tf.compat.v1.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # Maxpooling over the outputs
            self.h_pool = tf.squeeze(tf.nn.max_pool(self.h_pool, ksize=[1, 1, self.input_cnn.get_shape()[-2], 1], strides=[1, 1, 1, 1], padding='VALID', name="pool"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.final_output = tf.nn.dropout(self.h_pool, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W_output = tf.compat.v1.get_variable("W1", shape=[self.final_output.get_shape()[-1], 1], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            b_output = tf.Variable(tf.zeros([1]))
        self.scores = tf.compat.v1.nn.xw_plus_b(self.final_output, W_output, b_output, name="scores")
        self.predictions = tf.compat.v1.nn.sigmoid(self.scores)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.compat.v1.nn.softplus(self.scores * self.input_y)
            self.loss = tf.reduce_mean(losses)

        self.saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=500)

#positional embeddings
def add_positional_embedding(x, sequence_length, depth, name="pos"):
    with tf.name_scope("add_positional_embedding"):
        var = tf.cast(tf.compat.v1.get_variable(name, [sequence_length, depth], initializer=tf.contrib.layers.xavier_initializer(seed=1234)), x.dtype)
        return x + tf.expand_dims(var, 0)
