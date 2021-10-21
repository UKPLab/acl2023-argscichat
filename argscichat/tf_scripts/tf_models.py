import tensorflow as tf
import tensorflow_addons as tfa
from transformers import TFBertModel

from argscichat.utility.tensorflow_utils import get_initializer


class M_ComponentsSciBert(tf.keras.Model):

    def __init__(self, preloaded_name, config, label_list, **kwargs):
        super(M_ComponentsSciBert, self).__init__(**kwargs)
        self.label_list = label_list
        self.preloaded_name = preloaded_name
        self.config = config

        self.model = TFBertModel.from_pretrained(preloaded_name, from_pt=True)

        self.final_block = {label.name: tf.keras.layers.Dense(units=label.num_values,
                                                              kernel_initializer=get_initializer(
                                                                  0.02),
                                                              name="{}_classifier".format(label.name))
                            for label in self.label_list}

    def call(self, inputs, training=False, state='training', **kwargs):
        # [batch_size, dim]
        source_output = self.model({
            'input_ids': inputs['source_input_ids'],
            'attention_mask': inputs['source_attention_mask'],
        }, training=training)
        source_output = source_output['last_hidden_state'][:, 0, :]  # CLS token

        target_output = self.model({
            'input_ids': inputs['target_input_ids'],
            'attention_mask': inputs['target_attention_mask'],
        }, training=training)
        target_output = target_output['last_hidden_state'][:, 0, :]  # CLS token

        distance = tf.cast(inputs['distance'], target_output.dtype)

        logits = {
            # 'source_type': self.final_block['source_type'](source_output, training=training),
            # 'target_type': self.final_block['target_type'](target_output, training=training),
            'relation_type': self.final_block['relation_type'](
                tf.concat((source_output, target_output, distance), axis=-1),
                training=training)
        }

        # [batch_size, classes]
        return logits, {
            'raw_predictions': {key: tf.stop_gradient(tf.nn.softmax(value)) for key, value in logits.items()}}


class M_TokensSciBert(tf.keras.Model):

    def __init__(self, preloaded_name, config, label_list, **kwargs):
        super(M_TokensSciBert, self).__init__(**kwargs)
        self.label_list = label_list

        self.config = config

        self.model = TFBertModel.from_pretrained(preloaded_name, from_pt=True)

        self.final_block = {label.name: tf.keras.layers.Dense(units=label.num_values,
                                                              kernel_initializer=get_initializer(
                                                                  0.02),
                                                              name="{}_classifier".format(label.name))
                            for label in self.label_list}

    def call(self, inputs, training=False, state='training', **kwargs):
        tokens = inputs['input_ids']
        tokens_mask = tf.where(tokens == 0, tf.zeros_like(tokens), tf.ones_like(tokens))
        tokens_length = tf.reduce_sum(tokens_mask, axis=-1)

        # [batch_size, seq, dim]
        model_output = self.model({
            'input_ids': tokens,
            'attention_mask': inputs['attention_mask'],
        }, training=training)['last_hidden_state']

        logits = {key: block(model_output, training=training) for key, block in
                  self.final_block.items()}

        return logits, {
            'tokens_mask': tokens_mask,
            'tokens_length': tokens_length,
            'raw_predictions': {
                key: tf.stop_gradient(tf.nn.softmax(value) * tf.cast(tokens_mask, value.dtype)[:, :, None]) for
                key, value in logits.items()}}


class M_TokensSciBertCRF(M_TokensSciBert):

    def __init__(self, **kwargs):
        super(M_TokensSciBertCRF, self).__init__(**kwargs)

        self.crf = tfa.layers.CRF(units=self.label_list[0].num_values,
                                  chain_initializer='orthogonal',
                                  use_boundary=False,
                                  boundary_initializer='zeros',
                                  use_kernel=True)

    def call(self, inputs, training=False, state='training', **kwargs):
        tokens = inputs['input_ids']
        tokens_mask = tf.where(tokens == 0, tf.zeros_like(tokens), tf.ones_like(tokens))

        # [batch_size, seq, dim]
        model_output = self.model({
            'input_ids': tokens,
            'attention_mask': inputs['attention_mask'],
        }, training=training)['last_hidden_state']

        logits = self.final_block[self.label_list[0].name](model_output, training=training)
        decode_sequence, potentials, sequence_lengths, kernel = self.crf(logits)

        return {self.label_list[0].name: logits}, {
            'potentials': potentials,
            'sequence_lengths': sequence_lengths,
            'kernel': kernel,
            'tokens_mask': tokens_mask,
            'raw_predictions': {self.label_list[0].name: tf.stop_gradient(tf.nn.softmax(logits))}}
