import numpy as np
# from random import shuffle
# from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        example_dW = np.zeros(W.shape)
        for j in xrange(num_classes):
            if j == y[i]:
                example_dW[:, j] = (-X[i] *
                                    ((scores - correct_class_score + 1 > 0)
                                     .sum() - 1))
                continue
            example_dW[:, j] = X[i] * (scores[j] - correct_class_score + 1 > 0)

            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
        dW += example_dW

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    ###########################################################################
    # TODO:                                                                   #
    # Compute the gradient of the loss function and store it dW.              #
    # Rather that first computing the loss and then computing the derivative, #
    # it may be simpler to compute the derivative at the same time that the   #
    # loss is being computed. As a result you may need to modify some of the  #
    # code above to compute the gradient.                                     #
    ###########################################################################
    dW = (dW / float(num_train)) + (2 * reg * W)
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """

    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    ###########################################################################
    # TODO:                                                                   #
    # Implement a vectorized version of the structured SVM loss, storing the  #
    # result in loss.                                                         #
    ###########################################################################
    scores = X.dot(W)  # N x C
    correct_class_scores = (scores[np.arange(num_train), y].reshape((-1, 1)))
    loss = (scores - correct_class_scores + 1)
    loss[loss < 0] = 0
    loss = (loss.sum(axis=1) - 1).mean()
    loss += reg * np.sum(W * W)
    ###########################################################################
    #                                                      END OF YOUR CODE   #
    ###########################################################################

    ###########################################################################
    # TODO:                                                                   #
    # Implement a vectorized version of the gradient for the structured SVM   #
    # loss, storing the result in dW.                                         #
    #                                                                         #
    # Hint: Instead of computing the gradient from scratch, it may be easier  #
    # to reuse some of the intermediate values that you used to compute the   #
    # loss.                                                                   #
    ###########################################################################
    contributing_elements = ((scores - correct_class_scores + 1) > 0).astype('int')  # N x C
    assert np.all(contributing_elements.sum(axis=1) > 0)  # the correct class is delta (i.e., > 0)
    correct_class_contrib = contributing_elements.sum(axis=1) - 1

    # at the wrong class contribution is 1, at correct class contribution is -# of positive elements
    contributing_elements[np.arange(num_train), y] = -correct_class_contrib

    dW = X.T.dot(contributing_elements)  # D x C
    dW = (dW / float(num_train)) + (2 * reg * W)

    ###########################################################################
    #                                                       END OF YOUR CODE  #
    ###########################################################################

    return loss, dW
