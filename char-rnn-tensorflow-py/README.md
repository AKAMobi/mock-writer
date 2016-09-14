# char-rnn-tensorflow
Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

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

