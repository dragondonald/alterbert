# coding=utf-8

"""
BERT with reverse net
"""


from backpropagation import BPLayer, RevBPLayer
import tensorflow as tf
import copy
import math
import utils
import layers


class RevBert(object):
    """
    BERT model with reversible net, from <REFORMER: THE EFFICIENT TRANSFORMER>
    """

    def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):

        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
          input_ids: int32 Tensor of shape [batch_size, seq_length].
          input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
          use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings.
          scope: (optional) variable scope. Defaults to "revbert".

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = utils.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="revbert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_output_bplayer,
                 self.embedding_table, self.embedding_table_bplayer) = layers.embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output, self.embedding_output_bplayer = layers.embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob,
                    previor_bplayer=self.embedding_output_bplayer)

            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = utils.create_attention_mask_from_input_mask(
                    input_ids, input_mask)

                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    input_bplayer=self.embedding_output_bplayer,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=utils.get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output, self.sequence_output_bplayer = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler") as pooler_scope:
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=utils.create_initializer(config.initializer_range))
                self.pooled_output_bplayer = BPLayer(self.pooled_output, pooler_scope, [self.sequence_output_bplayer])

    def get_pooled_output(self):
        """
        Gets pooled output and the bplayer.
        Returns:
            Tuple
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the final hidden of the transformer encoder
            and the bplayer.
        """
        return self.pooled_output, self.pooled_output_bplayer

    def get_sequence_output(self):
        """
        Gets final hidden layer of encoder and the bplayer.
        Returns:
            Tuple
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the final hidden of the transformer encoder
            and the bplayer
        """
        return self.sequence_output, self.sequence_output_bplayer

    def get_all_encoder_layers(self):
        """
        Gets the list of all hidden layers of encoder and the their bplayers.
        Returns:
            list of tuple
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the final hidden of the transformer encoder
            and the bplayer
        """
        return self.all_encoder_layers

    def get_embedding_output(self):
        """
        Gets output of the embedding lookup (i.e., input to the transformer).
        Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the output of the embedding layer, after summing the word
            embeddings with the positional embeddings and the token type embeddings,
            then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output, self.embedding_output_bplayer

    def get_embedding_table(self):
        return self.embedding_table, self.embedding_table_bplayer


def transformer_model(input_tensor,
                      input_bplayer,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=utils.gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    """
    Multi-headed, multi-layer Transformer from "Attention is All You Need".
    This is almost an exact implementation of the original Transformer encoder.
    Add revnet.
    See the original paper:
    https://arxiv.org/abs/1706.03762
    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      input_bplayer: input bplayer
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.
    Returns:
        Tuple
        float Tensor of shape [batch_size, seq_length, hidden_size], the final
        hidden layer of the Transformer
        and bplayer
    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))
    with tf.variable_scope("transfomer_model"):
        with tf.variable_scope("prepare") as prepare_scope:
            attention_head_size = int(hidden_size / num_attention_heads)
            input_shape = utils.get_shape_list(input_tensor, expected_rank=3)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            input_width = input_shape[2]
            # The Transformer performs sum residuals on all layers so the input needs
            # to be the same as the hidden size.
            if input_width != hidden_size:
                raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                                 (input_width, hidden_size))
            # We keep the representation as a 2D tensor to avoid re-shaping it back and
            # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
            # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
            # help the optimizer.
            prev_output = utils.reshape_to_matrix(input_tensor) # [batch_size * seq_length, input_width]
            prev_bplayer = BPLayer(prev_output, prepare_scope, [input_bplayer])
        all_layer_outputs = []
        all_layer_bplayers = []
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output
                layer_output, bplayer = rev_transformer_layer(layer_input,
                                                              [prev_bplayer],
                                                     batch_size,
                                                     seq_length,
                                                     attention_head_size,
                                                     attention_mask,
                                                     num_attention_heads,
                                                     intermediate_size,
                                                     intermediate_act_fn,
                                                     hidden_dropout_prob,
                                                     attention_probs_dropout_prob,
                                                     initializer_range)
                all_layer_outputs.append(layer_output)
                prev_output = layer_output
                all_layer_bplayers.append(bplayer)
                prev_bplayer = bplayer
        with tf.variable_scope("output") as output_scope:
            if do_return_all_layers:
                if len(all_layer_outputs) != len(all_layer_bplayers):
                    raise Exception("transformer model: the num of all layer outputs is not equal to"
                                    "the num of all layer bplayers")
                final_outputs = []
                for i in range(len(all_layer_outputs)):
                    final_output = utils.reshape_from_matrix(all_layer_outputs[i], input_shape)
                    final_bplayer = BPLayer(final_output, output_scope, [all_layer_bplayers[i]])
                    final_outputs.append((final_output, final_bplayer))
                return final_outputs
            else:
                final_output = utils.reshape_from_matrix(prev_output, input_shape)
                final_bplayer = BPLayer(final_output, output_scope, [prev_bplayer])
                return (final_output, final_bplayer)


def rev_transformer_layer(input_tensor,
                                prev_bplayers,
                                batch_size,
                                seq_length,
                                attention_head_size,
                                attention_mask=None,
                                num_attention_heads=12,
                                intermediate_size=3072,
                                intermediate_act_fn=utils.gelu,
                                hidden_dropout_prob=0.1,
                                attention_probs_dropout_prob=0.1,
                                initializer_range=0.02,
                                exchange_output=True):
    """
    可逆残差网络层，将输入对半切分
    :param input_tensor:
    :param prev_bplayers:
    :param batch_size:
    :param seq_length:
    :param attention_head_size:
    :param attention_mask:
    :param num_attention_heads:
    :param intermediate_size:
    :param intermediate_act_fn:
    :param hidden_dropout_prob:
    :param attention_probs_dropout_prob:
    :param initializer_range:
    :param exchange_output:
    :return:
    """
    with tf.variable_scope("rev_transformer_layer") as layer_scope:
        x1, x2 = utils.split_for_rev(input_tensor)
        shape1 = utils.get_shape_list(x1)[-1]
        shape2 = utils.get_shape_list(x2)[-1]

        def create_func_f(attention_mask,
                hidden_size,
                hidden_dropout_prob,
                num_attention_heads,
                attention_head_size,
                attention_probs_dropout_prob,
                initializer_range,
                batch_size,
                seq_length):
            def func(input):
                return attention_func(input_tensor=input,
                               attention_mask=attention_mask,
                               hidden_size=hidden_size,
                               hidden_dropout_prob=hidden_dropout_prob,
                               num_attention_heads=num_attention_heads,
                               attention_head_size=attention_head_size,
                               attention_probs_dropout_prob=attention_probs_dropout_prob,
                               initializer_range=initializer_range,
                               batch_size=batch_size,
                               seq_length=seq_length)
            return func

        def create_func_g(intermediate_size,
                     initializer_range,
                     hidden_size,
                     hidden_dropout_prob,
                     intermediate_act_fn):
            def func(input):
                return feedforward_func(input_tensor=input,
                                      intermediate_size=intermediate_size,
                                      initializer_range=initializer_range,
                                      hidden_size=hidden_size,
                                      hidden_dropout_prob=hidden_dropout_prob,
                                      intermediate_act_fn=intermediate_act_fn)
            return func

        func_f = create_func_f(attention_mask=attention_mask,
                hidden_size=shape1,
                hidden_dropout_prob=hidden_dropout_prob,
                num_attention_heads=num_attention_heads,
                attention_head_size=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                batch_size=batch_size,
                seq_length=seq_length)
        func_g = create_func_g(intermediate_size=intermediate_size,
                     initializer_range=initializer_range,
                     hidden_size=shape2,
                     hidden_dropout_prob=hidden_dropout_prob,
                     intermediate_act_fn=intermediate_act_fn)
        f_x2, _ = func_f(x2)
        y1 = f_x2 + x1
        g_y1, _ = func_g(y1)
        y2 = g_y1 + x2
        if exchange_output:
            y = utils.concat_for_rev([y2, y1])
        else:
            y = utils.concat_for_rev([y1, y2])
    bplayer = RevBPLayer(y, layer_scope, func_f, func_g, exchange=exchange_output, backward_layers=prev_bplayers)
    return y, bplayer


def feedforward_func(input_tensor,
                     intermediate_size,
                     initializer_range,
                     hidden_size,
                     hidden_dropout_prob,
                     intermediate_act_fn=utils.gelu):
    # The activation is only applied to the "intermediate" hidden layer.
    with tf.variable_scope("feedforward") as scope:
        intermediate_output = tf.layers.dense(
            input_tensor,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=utils.create_initializer(initializer_range),
            name="intermediate_dense")
        # Down-project back to `hidden_size` then add the residual.
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=utils.create_initializer(initializer_range),
            name="intermediate_output")
        layer_output = utils.dropout(layer_output, hidden_dropout_prob)
        layer_output = utils.layer_norm(layer_output + input_tensor)
    return layer_output, scope


def attention_func(input_tensor,
                   attention_mask,
                   hidden_size,
                   hidden_dropout_prob,
                   num_attention_heads,
                   attention_head_size,
                   attention_probs_dropout_prob,
                   initializer_range,
                   batch_size,
                   seq_length):
    attention_heads = []
    with tf.variable_scope("attention") as scope:
        with tf.variable_scope("self"):
            attention_head = attention_layer(
                from_tensor=input_tensor,
                to_tensor=input_tensor,
                attention_mask=attention_mask,
                num_attention_heads=num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                do_return_2d_tensor=True,
                batch_size=batch_size,
                from_seq_length=seq_length,
                to_seq_length=seq_length)
            attention_heads.append(attention_head)
        if len(attention_heads) == 1:
            attention_output = attention_heads[0]
        else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = tf.concat(attention_heads, axis=-1)
        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
            attention_output = tf.layers.dense(
                attention_output,
                hidden_size,
                kernel_initializer=utils.create_initializer(initializer_range))
            attention_output = utils.dropout(attention_output, hidden_dropout_prob)
            attention_output = utils.layer_norm(attention_output + input_tensor)
    return attention_output, scope


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.
      This is an implementation of multi-headed attention based on "Attention
      is all you Need". If `from_tensor` and `to_tensor` are the same, then
      this is self-attention. Each timestep in `from_tensor` attends to the
      corresponding sequence in `to_tensor`, and returns a fixed-with vector.
      This function first projects `from_tensor` into a "query" tensor and
      `to_tensor` into "key" and "value" tensors. These are (effectively) a list
      of tensors of length `num_attention_heads`, where each tensor is of shape
      [batch_size, seq_length, size_per_head].
      Then, the query and key tensors are dot-producted and scaled. These are
      softmaxed to obtain attention probabilities. The value tensors are then
      interpolated by these probabilities, then concatenated back to a single
      tensor and returned.
      In practice, the multi-headed attention are done with transposes and
      reshapes rather than actual separate tensors.
      Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
          from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
          from_seq_length, to_seq_length]. The values should be 1 or 0. The
          attention scores will effectively be set to -infinity for any positions in
          the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
          attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
          * from_seq_length, num_attention_heads * size_per_head]. If False, the
          output will be of shape [batch_size, from_seq_length, num_attention_heads
          * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
          of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `to_tensor`.
      Returns:
        float Tensor of shape [batch_size, from_seq_length,
          num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
          true, this will be of shape [batch_size * from_seq_length,
          num_attention_heads * size_per_head]).
      Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
      """
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor
    from_shape = utils.get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = utils.get_shape_list(to_tensor, expected_rank=[2, 3])
    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")
    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")
    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    from_tensor_2d = utils.reshape_to_matrix(from_tensor)
    to_tensor_2d = utils.reshape_to_matrix(to_tensor)
    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=utils.create_initializer(initializer_range))
    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=utils.create_initializer(initializer_range))
    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=utils.create_initializer(initializer_range))
    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)
    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)
    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))
    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder
    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = utils.dropout(attention_probs, attention_probs_dropout_prob)
    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])
    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)
    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])
    return context_layer