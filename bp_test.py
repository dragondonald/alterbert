# -*- coding:utf-8 -*-


from backpropagation import BPLayer, RevBPLayer
import tensorflow as tf
import numpy as np
import random
import utils


def revbplayer_test():
    x = tf.placeholder(shape=[None, 2], dtype='float32')
    y_target = tf.placeholder(shape=[None], dtype='float32')

    def func_f(input):
        with tf.variable_scope("func_f") as f_scope:
            result = tf.layers.dense(input, units=1, activation=tf.nn.relu, kernel_initializer=tf.ones_initializer)
            return result, f_scope

    def func_g(input):
        with tf.variable_scope("func_g") as g_scope:
            result = tf.layers.dense(input, units=1, activation=tf.nn.relu, kernel_initializer=tf.ones_initializer)
            return result, g_scope

    def layer_revbp(input):
        with tf.variable_scope("layer_revbp") as layer_scope:
            x1, x2 = utils.split_for_rev(input)
            x1 = tf.Print(x1, [x1], message="network layer %s x1 === " % layer_scope.name)
            x2 = tf.Print(x2, [x2], message="network layer %s x2 === " % layer_scope.name)
            y1, _ = func_f(x2)
            y1 = tf.Print(y1, [y1], message="network layer %s f_x2 === " % layer_scope.name)
            y1 = y1 + x1
            y2, _ = func_g(y1)
            y2 = tf.Print(y2, [y2], message="network layer %s g_y1 === " % layer_scope.name)
            y2 = y2 + x2
            output = utils.concat_for_rev([y1, y2])
            output = tf.Print(output, [output], message="network layer %s y === " % layer_scope.name)
            revbplayer = RevBPLayer(output, layer_scope, func_f, func_g)
            return output, revbplayer

    def network(input, labels, num_layers):
        with tf.variable_scope("network") as network_scope:
            with tf.variable_scope("prepare") as prepare_scope:
                prev_output = input
                prev_bplayer = BPLayer(prev_output, prepare_scope)
            for i in range(num_layers):
                with tf.variable_scope("layer_%s" % i) as layernum_scope:
                    prev_output, bplayer = layer_revbp(prev_output)
                    bplayer.add_backward_layer(prev_bplayer)
                    prev_bplayer = bplayer
            with tf.variable_scope("loss") as loss_scope:
                y = tf.layers.dense(prev_output, units=1, kernel_initializer=tf.ones_initializer)
                y = tf.squeeze(y, axis=-1)
                loss = tf.losses.mean_squared_error(labels, y)
                bplayer = BPLayer(loss, loss_scope)
                bplayer.add_backward_layer(prev_bplayer)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_scope.name)
        return loss, bplayer, var_list, y

    with tf.variable_scope("bplayer_network") as bp_scope:
        loss_bp, bplayer_bp, _, _ = network(x, y_target, 4)
        opt_bp = tf.train.GradientDescentOptimizer(0.01)
        with tf.variable_scope("backward_gradients"):
            grads_vals_bp = bplayer_bp.backward_gradients()
        train_op_bp = opt_bp.apply_gradients(grads_vals_bp)
        grads_bp, vals_bp, grad_names_bp, val_names_bp = separate_grads_vals(grads_vals_bp)
    with tf.variable_scope("origin_network") as origin_scope:
        loss_origin, _, var_list_origin, _ = network(x, y_target, 4)
        opt_origin = tf.train.GradientDescentOptimizer(0.01)
        grads_vals_origin = opt_origin.compute_gradients(loss_origin, var_list_origin)
        train_op_origin = opt_origin.apply_gradients(grads_vals_origin)
        grads_origin, vals_origin, grad_names_origin, val_names_origin = separate_grads_vals(grads_vals_origin)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 判断初始化是否一致
        x_input = (random.random() * 10, random.random() * 10)
        y_input = x_input[0] * x_input[0] * 3 + x_input[0] * x_input[1] * 2 + x_input[1] * x_input[1] + 5
        result_bp = sess.run(vals_bp,
                             feed_dict={x: (x_input,), y_target: (y_input,)})
        result_origin = sess.run(vals_origin,
                                 feed_dict={x: (x_input,), y_target: (y_input,)})
        for i in range(len(val_names_bp)):
            name_bp = val_names_bp[i]
            var_bp = result_bp[i]
            name_origin = name_bp.replace("bplayer_network", "origin_network")
            if name_origin in val_names_origin:
                var_origin = result_origin[val_names_origin.index(name_origin)]
                assert np.array_equal(var_bp, var_origin), "init var(%s) not equal, bp=%s, origin=%s" % (
                name_bp, var_bp, var_origin)
            else:
                raise Exception("init origin has no var(%s)" % name_origin)
        # 判断梯度更新是否一致
        for i in range(1):
            x_input = (random.random() * 10, random.random() * 10)
            y_input = x_input[0] * x_input[0] * 3 + x_input[0] * x_input[1] * 2 + x_input[1] * x_input[1] + 5
            result_bp = sess.run([train_op_bp, loss_bp] + grads_bp + vals_bp, feed_dict={x: (x_input, ), y_target: (y_input, )})
            names_vars_bp = vars_dict(grad_names_bp, val_names_bp, result_bp[2:])
            result_origin = sess.run([train_op_origin, loss_origin] + grads_origin + vals_origin, feed_dict={x: (x_input, ), y_target: (y_input, )})
            names_vars_origin = vars_dict(grad_names_origin, val_names_origin, result_origin[2:])
            assert np.array_equal(result_origin[1], result_bp[1]), "epoch %s loss not equal, bp=%s, origin=%s" % (i, result_bp[1], result_origin[1])
            for name_bp, (grad_result_bp, val_result_bp) in names_vars_bp.items():
                name_origin = name_bp.replace("bplayer_network", "origin_network")
                if name_origin in names_vars_origin:
                    grad_result_origin, val_result_origin = names_vars_origin[name_origin]
                    print("epoch %s grad(%s) bp=%s, origin=%s" % (i, name_bp, grad_result_bp, grad_result_origin))
                    print("epoch %s val(%s) bp=%s, origin=%s" % (i, name_bp, val_result_bp, val_result_origin))

                    assert np.average(np.subtract(grad_result_bp, grad_result_origin)) < 0.001, "epoch %s grad(%s) not equal, bp=%s, origin=%s" % (i, name_bp, grad_result_bp, grad_result_origin)
                    assert np.average(np.subtract(val_result_bp, val_result_origin)) < 0.001, "epoch %s val(%s) not equal, bp=%s, origin=%s" % (i, name_bp, val_result_bp, val_result_origin)
                    # assert np.array_equal(grad_result_bp, grad_result_origin), "epoch %s grad(%s) not equal, bp=%s, origin=%s" % (i, name_bp, grad_result_bp, grad_result_origin)
                    # assert np.array_equal(val_result_bp, val_result_origin), "epoch %s val(%s) not equal, bp=%s, origin=%s" % (i, name_bp, val_result_bp, val_result_origin)
                else:
                    raise Exception("origin has no var(%s)" % name_origin)


def bplayer_test2():
    x = tf.placeholder(shape=[2], dtype='float32')
    y_target = tf.placeholder(shape=[1], dtype='float32')

    def network(inputs, targets):
        with tf.variable_scope("network") as scope:
            with tf.variable_scope("w") as scope_w:
                w = tf.get_variable("w", [1, 2], dtype=tf.float32, initializer=tf.ones_initializer)
                bplayer_w = BPLayer(w, scope_w)
            with tf.variable_scope("add") as scope_calc:
                b = tf.get_variable("b", [1], dtype=tf.float32, initializer=tf.zeros_initializer)
                y = tf.matmul(w, tf.expand_dims(inputs, axis=-1)) + b
                y = tf.squeeze(y, axis=-1)
                loss = tf.losses.mean_squared_error(targets, y)
                loss = tf.reduce_mean(loss)
                bplayer = BPLayer(loss, scope_calc, [bplayer_w])
            return loss, bplayer, [w, b], y

    with tf.variable_scope("bplayer_network") as bp_scope:
        loss_bp, bplayer_bp, _, _ = network(x, y_target)
        opt_bp = tf.train.GradientDescentOptimizer(0.01)
        with tf.variable_scope("backward_gradients"):
            grads_vals_bp = bplayer_bp.backward_gradients()
        train_op_bp = opt_bp.apply_gradients(grads_vals_bp)
        grads_bp, vals_bp, grad_names_bp, val_names_bp = separate_grads_vals(grads_vals_bp)
    with tf.variable_scope("origin_network") as origin_scope:
        loss_origin, _, var_list_origin, _ = network(x, y_target)
        opt_origin = tf.train.GradientDescentOptimizer(0.01)
        grads_vals_origin = opt_origin.compute_gradients(loss_origin, var_list_origin)
        train_op_origin = opt_origin.apply_gradients(grads_vals_origin)
        grads_origin, vals_origin, grad_names_origin, val_names_origin = separate_grads_vals(grads_vals_origin)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1):
            x_input = (random.random() * 10, random.random() * 10)
            y_input = (10 * x_input[0] + 5 * x_input[1] + 3, )
            result_bp = sess.run([train_op_bp, loss_bp] + grads_bp + vals_bp, feed_dict={x: x_input, y_target: y_input})
            names_vars_bp = vars_dict(grad_names_bp, val_names_bp, result_bp[2:])
            result_origin = sess.run([train_op_origin, loss_origin] + grads_origin + vals_origin, feed_dict={x: x_input, y_target: y_input})
            names_vars_origin = vars_dict(grad_names_origin, val_names_origin, result_origin[2:])
            for name_bp, (grad_result_bp, val_result_bp) in names_vars_bp.items():
                name_origin = name_bp.replace("bplayer_network", "origin_network")
                if name_origin in names_vars_origin:
                    grad_result_origin, val_result_origin = names_vars_origin[name_origin]
                    print(grad_result_origin)
                    print(val_result_origin)
                    assert np.array_equal(grad_result_bp, grad_result_origin), "epoch %s grad(%s) not equal, bp=%s, origin=%s" % (i, name_bp, grad_result_bp, grad_result_origin)
                    assert np.array_equal(val_result_bp, val_result_origin), "epoch %s val(%s) not equal, bp=%s, origin=%s" % (i, name_bp, val_result_bp, val_result_origin)
                else:
                    raise Exception("origin has no var(%s)" % name_origin)


def bplayer_test1():
    with tf.variable_scope('b') as b_scope:
        w1 = tf.get_variable('w1', shape=[1], initializer=tf.ones_initializer)
        bplayer_b = BPLayer(w1, b_scope)

    with tf.variable_scope('c') as c_scope:
        w2 = tf.get_variable('w2', shape=[1], initializer=tf.ones_initializer)
        y = w1 + w2
        bplayer = BPLayer(y, c_scope, [bplayer_b])

    trainable_vars_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, b_scope.name)
    trainable_vars_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, c_scope.name)
    print(trainable_vars_b)
    print(trainable_vars_c)

    with tf.variable_scope("bplayer_network") as bp_scope:
        opt_bp = tf.train.GradientDescentOptimizer(0.01)
        with tf.variable_scope("backward_gradients"):
            grads_vals_bp = bplayer.backward_gradients()
        train_op_bp = opt_bp.apply_gradients(grads_vals_bp)
        grads_bp, vals_bp, grad_names_bp, val_names_bp = separate_grads_vals(grads_vals_bp)
        print(val_names_bp)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result_origin = sess.run([train_op_bp] + grads_bp + vals_bp)
        print(result_origin[1:])


def bplayer_test():
    x = tf.placeholder(shape=[2], dtype='float32')
    y_target = tf.placeholder(shape=[1], dtype='float32')

    def network(inputs, targets):
        with tf.variable_scope("network") as scope:
            w = tf.get_variable("w", [1, 2], dtype=tf.float32, initializer=tf.ones_initializer)
            b = tf.get_variable("b", [1], dtype=tf.float32, initializer=tf.zeros_initializer)
            y = tf.matmul(w, tf.expand_dims(inputs, axis=-1)) + b
            y = tf.squeeze(y, axis=-1)
            loss = tf.losses.mean_squared_error(targets, y)
            loss = tf.reduce_mean(loss)
            bplayer = BPLayer(loss, scope)
            return loss, bplayer, [w, b], y

    with tf.variable_scope("bplayer_network") as bp_scope:
        loss_bp, bplayer_bp, _, _ = network(x, y_target)
        opt_bp = tf.train.GradientDescentOptimizer(0.01)
        with tf.variable_scope("backward_gradients"):
            grads_vals_bp = bplayer_bp.backward_gradients()
        train_op_bp = opt_bp.apply_gradients(grads_vals_bp)
        grads_bp, vals_bp, grad_names_bp, val_names_bp = separate_grads_vals(grads_vals_bp)
    with tf.variable_scope("origin_network") as origin_scope:
        loss_origin, _, var_list_origin, _ = network(x, y_target)
        opt_origin = tf.train.GradientDescentOptimizer(0.01)
        grads_vals_origin = opt_origin.compute_gradients(loss_origin, var_list_origin)
        train_op_origin = opt_origin.apply_gradients(grads_vals_origin)
        grads_origin, vals_origin, grad_names_origin, val_names_origin = separate_grads_vals(grads_vals_origin)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            x_input = (random.random() * 10, random.random() * 10)
            y_input = (10 * x_input[0] + 5 * x_input[1] + 3, )
            result_bp = sess.run([train_op_bp, loss_bp] + grads_bp + vals_bp, feed_dict={x: x_input, y_target: y_input})
            names_vars_bp = vars_dict(grad_names_bp, val_names_bp, result_bp[2:])
            result_origin = sess.run([train_op_origin, loss_origin] + grads_origin + vals_origin, feed_dict={x: x_input, y_target: y_input})
            names_vars_origin = vars_dict(grad_names_origin, val_names_origin, result_origin[2:])
            for name_bp, (grad_result_bp, val_result_bp) in names_vars_bp.items():
                name_origin = name_bp.replace("bplayer_network", "origin_network")
                if name_origin in names_vars_origin:
                    grad_result_origin, val_result_origin = names_vars_origin[name_origin]
                    assert np.array_equal(grad_result_bp, grad_result_origin), "epoch %s grad(%s) not equal, bp=%s, origin=%s" % (i, name_bp, grad_result_bp, grad_result_origin)
                    assert np.array_equal(val_result_bp, val_result_origin), "epoch %s val(%s) not equal, bp=%s, origin=%s" % (i, name_bp, val_result_bp, val_result_origin)
                else:
                    raise Exception("origin has no var(%s)" % name_origin)


def separate_grads_vals(grads_vals):
    grads, vals = zip(*grads_vals)
    grad_names = [t.name for t in grads]
    val_names = [t.name for t in vals]
    return list(grads), list(vals), grad_names, val_names


def vars_dict(grad_names, val_names, results):
    """

    :param grad_names:
    :param val_names:
    :param results:
    :return: (grad_result, val_result)
    """
    kv = {}
    i = 0
    num_vars = len(grad_names)
    for name in val_names:
        kv[name] = (results[i], results[i + num_vars])
        i += 1
    return kv


if __name__ == '__main__':
    bplayer_test1()

