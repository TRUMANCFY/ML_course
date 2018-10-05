import numpy as np


def compute_gradient(y, tx, w):
    """compute the gradient"""
    error = y - tx.dot(w)
    grad = -tx.T.dot(error) / len(error)
    return error, grad


def compute_mse(error):
    return np.mean(np.square(error))


def calculate_mse(y, tx, w):
    error = y - tx.dot(w)
    return compute_mse(error)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffled_indexes = np.random.permutation(np.arrange(data_size))
        shuffled_y = y[shuffled_indexes]
        shuffled_tx = tx[shuffled_indexes]

    else:
        shuffled_y = y
        shuffled_tx = x
    
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min(data.size, batch_size * (batch_num + 1))
        if start_index != end_index:
            yield shuffled_tx, shuffled_y

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear Regression using Gradient Descent"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for i in range(max_iters):
        error, grad = compute_gradient(y, tx, w)
        loss = compute_mse(error)
        w = w - grad * gamma
        ws.append(w)
        losses.append(loss)
    
        print('GD: {iter} / {max_iters}: loss={l}, w={w}'.format(iter=i, max_iters=max_iters, l=loss, w=w))
    
    return ws, losses
    

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for i in range(max_iters):
        for batch_x, batch_y in batch_iter(y=y, tx=tx, batch_size=1, num_batches=len(y), shuffle=True):
            error, grad = compute_gradient(batch_x, batch_y, w)
            w = w - grad * gamma
            loss = calculate_mse(y, tx, w)
            ws.append(w)
            losses.append(loss)
            
        print('SGD: {iter} / {max_iters}: loss={l}, w={w}'.format(iter=i, max_iters=max_iters, l=loss, w=w))
    
    return ws, losses


def least_squares(y, tx):
    n, _ = np.shape(tx)
    w = np.matmul(np.linalg.inv(np.matmul((tx.T), tx)), tx.T).dot(y)
    error = np.sum(np.square(y - tx.dot(w))) / (2*n)
    return error, w


def ridge_regression(y, tx, lambda_):
    return np.linalg.inv(tx.T.dot(tx) + (2*tx.shape[0])*lambda_*np.identity(tx.shape[1])).dot(tx.T).dot(y)

def logistic_regression(y, x, initial_w, max_iters, gamma):
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError