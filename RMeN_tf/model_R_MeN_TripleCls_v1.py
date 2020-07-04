import tensorflow as tf
from sonnet.python.modules import relational_memory
import math


class RMeN(object):

    def __init__(self, vocab_size, embedding_size, batch_size, initialization, mem_slots, num_heads,
                 use_pos, attention_mlp_layers, head_size):
        # Placeholders for input, output
        self.input_x = tf.placeholder(tf.int32, [batch_size, 3], name="input_h")
        self.input_y = tf.placeholder(tf.float32, [batch_size, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("embedding"):
            if initialization != []:
                self.input_feature = tf.get_variable(name="input_feature_1", initializer=initialization)
            else:
                self.input_feature = tf.get_variable(name="input_feature_2", shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=1234))

        # Embedding lookup
        self.emb = tf.nn.embedding_lookup(self.input_feature, self.input_x)

        if use_pos == 1:
            self.emb = add_positional_embedding(self.emb, 3, embedding_size)

        self.h_emb, self.r_emb, self.t_emb = tf.split(self.emb, num_or_size_splits=3, axis=1)

        self.h_emb = tf.squeeze(self.h_emb)
        self.r_emb = tf.squeeze(self.r_emb)
        self.t_emb = tf.squeeze(self.t_emb)

        gen_mem = relational_memory.RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads,
                                                      gate_style='memory', attention_mlp_layers=attention_mlp_layers)

        init_states = gen_mem.initial_state(batch_size=batch_size)

        mem_output1, memory_input_next_step = gen_mem(self.h_emb, init_states)
        mem_output2, memory_input_next_step = gen_mem(self.r_emb, memory_input_next_step)
        mem_output3, memory_input_next_step = gen_mem(self.t_emb, memory_input_next_step)

        self.final_output = tf.nn.dropout(mem_output1 * mem_output2 * mem_output3, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output1"):
            W1 = tf.get_variable("W1", shape=[self.final_output.get_shape()[-1], 1], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            b1 = tf.Variable(tf.zeros([1]))

        self.scores = tf.nn.xw_plus_b(self.final_output, W1, b1, name="scores")
        self.predictions = tf.nn.sigmoid(self.scores)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softplus(self.scores * self.input_y)
            self.loss = tf.reduce_mean(losses)

        self.saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=500)


def add_positional_embedding(x, sequence_length, depth, name="pos"):
    with tf.name_scope("add_positional_embedding"):
        var = tf.cast(tf.get_variable(name, [sequence_length, depth], initializer=tf.contrib.layers.xavier_initializer(seed=1234)), x.dtype)
        return x + tf.expand_dims(var, 0)

