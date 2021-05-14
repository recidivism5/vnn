I'm looking to make a meme generator and a chatbot. Right now though, in this repository, vectorized_nn.c is an improved, vectorized version of the neural network generator/trainer I made for "number recognizer".

vectorized_nn.c has a better command-line-interface as well. As usual, you don't need anything but this repository and a good C compiler to generate a 28x28 digit image recognizer and train it however you want.


command format:

./vectorized_nn.exe <*.network> <-train / -test> <numEpochs / testDataStartIndex> <learningRate / numTestingPairs>


Apart from that, I also have vec_nn_ad.c which is the vectorized neural network generator/trainer, except it implements a slightly altered backpropagation routine which is described in a comment at the top of its source code.