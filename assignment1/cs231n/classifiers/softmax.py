import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train, num_feat = X.shape
    num_class = W.shape[1]

    def sm(x):
        y = x.copy()
        y -= np.max(y)
        return np.exp(y) / np.exp(y).sum()


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in xrange(num_train):
        x = X[i]
        f = np.zeros((num_class,))
        for j in xrange(num_class):
            f[j] = np.dot(x, W[:,j])

        s = sm(f)

        example_dW = np.zeros(W.shape)
        for j in xrange(num_class):
            example_dW[:,j] = (X[i] * (s[j] - float(y[i] == j)))

        loss -= np.log(s[y[i]])
        dW += example_dW

    loss = (loss / float(num_train)) + (reg * np.sum(W*W))
    dW   = (dW / float(num_train)) + (2 * reg * W)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train, num_feat = X.shape
    num_class = W.shape[1]

    def sm(x):
        y = x.copy()
        y -= np.max(y)
        return np.exp(y) / np.exp(y).sum()

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    F = X.dot(W)   # F_{ij} = x_i^TW_j, shape: num_train x num_class (NxC)
    F -= np.max(F, axis=1).reshape((-1, 1))
    Y = F[np.array(range(num_train)), y]
    loss = -np.log((np.exp(Y) / np.exp(F).sum(axis=1))).mean() + (reg * np.sum(W*W))

    S = np.exp(F) / np.exp(F).sum(axis=1).reshape((-1, 1))  # N x C
    A = X.T.dot(S)
    B = np.zeros(W.shape)
    for i in xrange(num_class):
        B[:,i] = X[np.where(y == i)].sum(axis=0)

    #for i in xrange(num_train):
    #    B[:,y[i]] += X[i]
    dW = A - B

    dW = (dW / float(num_train)) + (2 * reg * W)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

