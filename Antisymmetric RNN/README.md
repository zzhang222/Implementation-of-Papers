# Pytorch implementation of AntisymmetricRNN
This is an implementation of [Antisymmetric RNN](https://openreview.net/pdf?id=ryxepo0cFX)
paper using the Pytorch-based framework [Learner](https://github.com/jpzxshi/learner). I also used codes from [this repo](https://github.com/hsjeong5/MNIST-for-Numpy) to get MNIST images in the format of numpy.

To run the example of Pixel MNIST:
```
python pixel.py
```

To compare the results with other baseline methods like LSTM, GRU and Vanilla RNN, change the variable `net_type` to corresponding method names like `LSTM`, `GRU`, `RNN`. If you'd like to add more RNN architectures by yourself, you can simply add them in the `rnn.py` file in `ASNN/learner/nn`.

## Slides
* PDF [Paper Review](Antisymmetric_RNN.pdf)


