# char-rnn-tensorflow
Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# How to use

1. Install pip (or pip3 for python3) if it is not already installed

   ```shell
   # Ubuntu/Linux 64-bit
   $ sudo apt-get install python-pip python-dev

   # Mac OS X
   $ sudo easy_install pip
   $ sudo easy_install --upgrade six
   ```

   ​

2. Install Tensorflow

   * Select the correct binary to install

     ```shell
     # Ubuntu/Linux 64-bit, CPU only, Python 2.7
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl

     # Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
     # Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl

     # Mac OS X, CPU only, Python 2.7:
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py2-none-any.whl

     # Mac OS X, GPU enabled, Python 2.7:
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.10.0rc0-py2-none-any.whl

     # Ubuntu/Linux 64-bit, CPU only, Python 3.4
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp34-cp34m-linux_x86_64.whl

     # Ubuntu/Linux 64-bit, GPU enabled, Python 3.4
     # Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp34-cp34m-linux_x86_64.whl

     # Ubuntu/Linux 64-bit, CPU only, Python 3.5
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl

     # Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
     # Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl

     # Mac OS X, CPU only, Python 3.4 or 3.5:
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py3-none-any.whl

     # Mac OS X, GPU enabled, Python 3.4 or 3.5:
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.10.0rc0-py3-none-any.whl
     ```

     ​

   * Install TensorFlow

     ```shell
     # Python 2
     $ sudo pip install --upgrade $TF_BINARY_URL

     # Python 3
     $ sudo pip3 install --upgrade $TF_BINARY_URL
     ```

     ​

3. Prepare train text

   Put the train text `input.txt`  in the `data\tinyshakespeare` folder.

   ​

4. Train model

   `--rnn_size=300 --model=gru --num_layers=2` maybe can get the better model.

5. Sample

   ​

# Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`.

To sample from a checkpointed model, `python sample.py`.

# Advanced Usage

To train with Advanced parameters on the tinyshakespeare corpus, run 

```shell
usage: train.py [-h] [--data_dir DATA_DIR] [--save_dir SAVE_DIR]
                [--rnn_size RNN_SIZE] [--num_layers NUM_LAYERS]
                [--model MODEL] [--batch_size BATCH_SIZE]
                [--seq_length SEQ_LENGTH] [--num_epochs NUM_EPOCHS]
                [--save_every SAVE_EVERY] [--grad_clip GRAD_CLIP]
                [--learning_rate LEARNING_RATE] [--decay_rate DECAY_RATE]
                [--init_from INIT_FROM]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   data directory containing input.txt
  --save_dir SAVE_DIR   directory to store checkpointed models
  --rnn_size RNN_SIZE   size of RNN hidden state
  --num_layers NUM_LAYERS
                        number of layers in the RNN
  --model MODEL         rnn, gru, or lstm
  --batch_size BATCH_SIZE
                        minibatch size
  --seq_length SEQ_LENGTH
                        RNN sequence length
  --num_epochs NUM_EPOCHS
                        number of epochs
  --save_every SAVE_EVERY
                        save frequency
  --grad_clip GRAD_CLIP
                        clip gradients at this value
  --learning_rate LEARNING_RATE
                        learning rate
  --decay_rate DECAY_RATE
                        decay rate for rmsprop
  --init_from INIT_FROM
                        continue training from saved model at this path. Path
                        must contain files saved by previous training process:
                        'config.pkl' : configuration; 'chars_vocab.pkl' :
                        vocabulary definitions; 'checkpoint' : paths to model
                        file(s) (created by tf). Note: this file contains
                        absolute paths, be careful when moving files around;
                        'model.ckpt-*' : file(s) with model definition
                        (created by tf)
```

