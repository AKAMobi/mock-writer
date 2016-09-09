import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        # 这里的infer被默认为False，只有在测试效果
        # 的时候才会被设计为True，在True的状态下
        # 只有一个batch，time step也被设计为1,我们
        # 可以由此观测训练成功
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        # 这里是选择RNN cell的类型，备选的有lstm, gru和simple rnn
        # 这里由输入的arg里的model参数作为测试标准，默认为lstm
        # 但是，我们可以看到，这里通过不同的模型我们可以用不同
        # 的cell。
        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # 定义cell的神经元数量，等同于cell = rnn_cell.BasicLSTMCell(args.rnn_size)
        cell = cell_fn(args.rnn_size)

        # 由于结构为多层结构，我们运用MultiRNNCell来定义神经元层。
        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        # 输入，同PTB模型，输入的格式为batch_size X sequence_length(step)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            # 使用GPU训练神经网络
            with tf.device('/gpu:0'):
                # 这里运用embedding来将输入的不同词汇map到隐含层的神经元上
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                # 这里对input的shaping很有意思。这个地方如果我们仔细去读PTB模型就会发现在他的
                # outputs = []这行附近有一段注释的文字，解释了一个alternative做法，这个做法就是那
                # alternative的方法。首先，我们将embedding_loopup所得到的[batch_size, seq_length, rnn_size]
                # tensor按照sequence length划分为一个list的[batch_size, 1, rnn_size]的tensor以表示每个
                # 步骤的输入。之后通过squeeze把那个1维度去掉，达成一个list的[batch_size, rnn_size]
                # 输入来被我们的rnn模型运用。
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # 这里定义的loop实际在于当我们要测试运行结果，即让机器自己写文章时，我们需要对每一步
        # 的输出进行查看。如果我们是在训练中，我们并不需要这个loop函数。
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # 这里我们得益于tensorflow强大的内部函数，rnn_decoder可以作为黑盒子直接运用，省去了编写
        # 的麻烦。另外，上面的loop函数只有在infer是被定为true的时候才会启动，一如我们刚刚所述。另外
        # rnn_decoder在tensorflow中的建立方式是以schedule sampling算法为基础制作的，故其自身已经融入
        # 了schedule sampling算法。
        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')、
        # 这里的过程可以说基本等同于PTB模型，首先通过对output的重新梳理得到一个
        # [batch_size*seq_length, rnn_size]的输出，并将之放入softmax里，并通过sequence
        # loss by example函数进行训练。
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
