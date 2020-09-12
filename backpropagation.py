# -*- coding:utf-8 -*-


import tensorflow as tf
import utils


class BPLayer(object):
    """
    计算梯度，并向后传递
    """

    def __init__(self, output, variable_scope, backward_layers=None):
        """

        :param output: scope最后的输出tensor
        :param variable_scope: 待计算梯度的scope
        :param backward_layers: 需要传递梯度的BPLayers
        """
        self.output = output
        self.bp_scope_name = "bp/" + variable_scope.name
        self.variable_scope = variable_scope
        if backward_layers is None:
            self.backward_layers = []
        else:
            self.backward_layers = backward_layers

    def grad_calc(self, y, dy, variables=None):
        with tf.variable_scope(self.bp_scope_name):
            grads = tf.gradients(y, variables, dy, gate_gradients=True)
        return list(zip(grads, variables))

    def add_backward_layer(self, backward_layer):
        if isinstance(backward_layer, BPLayer):
            self.backward_layers.append(backward_layer)
        else:
            raise Exception("BPLayer add backward layer error: the layer(%s) is not a BPLayer" % str(type(backward_layer)))

    def backward_gradients(self, dy=None):
        """
        计算梯度，并向后传递
        :param dy:grad_ys, When `grad_ys` is None,we fill in a tensor of '1's of the shape of y for each y in `ys`.
        :return:
        """
        backward_layers_num = len(self.backward_layers)
        # 需要计算梯度的变量
        variables = []
        # 需要向后传递梯度的BPLayers的output
        for layer in self.backward_layers:
            variables.append(layer.get_output())
        # scope中TRAINABLE_VARIABLES
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.variable_scope.name)
        variables.extend(trainable_vars)
        grad_vals = self.grad_calc(self.output, dy, variables)
        # 向后传递的梯度值
        backward_grads = [grad_val[0] for grad_val in grad_vals[0:backward_layers_num]]
        weight_grad_vals = grad_vals[backward_layers_num:]
        # 递归计算梯度
        i = 0
        for layer in self.backward_layers:
            weight_grad_vals.extend(layer.backward_gradients(backward_grads[i]))
            i += 1
        return weight_grad_vals

    def get_output(self):
        return self.output


class RevBPLayer(BPLayer):
    """
   用于可逆残差网络层计算梯度
    """

    def __init__(self, output, variable_scope, func_f, func_g, exchange=True, backward_layers=None):
        """

        :param output:
        :param variable_scope:
        :param func_f:
        :param func_g:
        :param exchange: y1和y2合并时前后顺序是否调换
        :param backward_layers:
        """
        super(RevBPLayer, self).__init__(output, variable_scope, backward_layers)
        self.func_f = func_f
        self.func_g = func_g
        self.exchange = True

    def grad_calc(self, y, dy, variables=None):
        with tf.variable_scope(self.bp_scope_name):
            y_ = tf.stop_gradient(y)
            y1, y2 = utils.split_for_rev(y_)
            if dy is None:
                dy1_ = tf.ones_like(y1)
                dy2_ = tf.ones_like(y2)
            else:
                dy1_, dy2_ = utils.split_for_rev(dy)
            if self.exchange:
                dy1 = dy2_
                dy2 = dy1_
            else:
                dy1 = dy1_
                dy2 = dy2_
            # 还原输入
            with tf.variable_scope(self.variable_scope, reuse=True):
                g_y1, scope_g = self.func_g(y1)
            x2 = y2 - g_y1
            with tf.variable_scope(self.variable_scope, reuse=True):
                f_x2, scope_f = self.func_f(x2)
            x1 = y1 - f_x2

            # z = f(x2) + x1, y2 = g(z) + x2, y1 = z
            # dz = dy1 + dy2_dz, dx1 = dz, dx2 = dy2 + dz_dx2
            weights_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_g.name)
            weights_f = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_f.name)
            grads_g = tf.gradients(g_y1, [y1] + weights_g, dy2)
            dy2_dz = grads_g[0]
            grads_weights_g = grads_g[1:]
            dz = dy1 + dy2_dz
            dx1 = dz
            grads_f = tf.gradients(f_x2, [x2] + weights_f, dz)
            dz_dx2 = grads_f[0]
            grads_weights_f = grads_f[1:]
            dx2 = dy2 + dz_dx2
            dx = utils.concat_for_rev([dx1, dx2])
            x = utils.concat_for_rev([x1, x2])
            variables = [x] + weights_f + weights_g
            grads = [dx] + grads_weights_f + grads_weights_g
        return list(zip(grads, variables))

    def add_backward_layer(self, backward_layer):
        # 可逆残差层只有一个输入，所以只有一个需要向后传递梯度的层
        if len(self.backward_layers) == 1:
            raise Exception("RevBPlayer has one backward layer, can not add another")
        if isinstance(backward_layer, BPLayer):
            self.backward_layers.append(backward_layer)
        else:
            raise Exception("BPLayer add backward layer error: the layer(%s) is not a BPLayer" % str(type(backward_layer)))

    def backward_gradients(self, dy=None):
        grad_vals = self.grad_calc(self.output, dy)
        dx, x = grad_vals[0]
        weight_grad_vals = grad_vals[1:]
        for layer in self.backward_layers:
            if isinstance(layer, RevBPLayer):
                # 如果向后传递的层是可逆残差层，将还原的x作为该层的output
                layer.set_output(x)
            weight_grad_vals.extend(layer.backward_gradients(dx))
        return weight_grad_vals

    def set_output(self, output):
        self.output = output


