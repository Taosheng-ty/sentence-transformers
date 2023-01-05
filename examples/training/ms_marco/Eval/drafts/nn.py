import numpy as np
import pickle
def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))
# x=np.array([1,1,1])
# b1=np.array([-1,1])
# w1=np.array([[-2,2],[-4,3]])

# b2=np.array([-1,1])
# w2=np.array([[-2,2],[-3,3]])

# w3=np.array([-1])
# b3=np.array([2,-1.5])




def affine_forward(x, w, b):

    out=x@w+b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache

    shape=np.array(x.shape)
    db=np.sum(dout,0)
    dw=x.T@dout
    dx=np.reshape(dout@w.T,shape)

    return dx, dw, db
  
def SigmoidFoward(x):
  output=sigmoid(x)
  cache=(output)
  return output,cache
def SigmoidBackward(dout, cache):
  sigma=cache[0]
  dsigma=sigma*(1-sigma)*dout
  return dsigma

def affine_sigmoid_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = SigmoidFoward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_sigmoid_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = SigmoidBackward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db
  
def mse_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  loss=np.mean(0.5*(y-x)**2)
  dx=-(y-x)
  return loss, dx


class ThreeLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, outputDim=1,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}

        
        self.reg=reg
        W1=np.random.normal(0.0,weight_scale,(input_dim,hidden_dim))
        b1=np.zeros(hidden_dim)
        W2=np.random.normal(0.0,weight_scale,(hidden_dim,hidden_dim))
        b2=np.zeros(hidden_dim)        
        W3=np.random.normal(0.0,weight_scale,(hidden_dim,outputDim))
        b3=np.zeros(outputDim) 
        self.params["W1"]=W1
        self.params["W2"]=W2
        self.params["W3"]=W3
        self.params["b1"]=b1
        self.params["b2"]=b2
        self.params["b3"]=b3


    def setWeight(self):
        self.params["b1"]=np.array([-1,1])
        self.params["W1"]=np.array([[-2,2],[-4,3]])

        self.params["b2"]=np.array([-1,1])
        self.params["W2"]=np.array([[-2,2],[-3,3]])

        self.params["W3"]=np.array([[2,-1.5]]).T
        self.params["b3"]=np.array([-1])
      
      
    def loss(self, X, y=None):

        out, cache1 = affine_sigmoid_forward(X, self.params["W1"], self.params["b1"])
        out2, cache2=affine_sigmoid_forward(out, self.params["W2"], self.params["b2"])
        out3, cache3=affine_forward(out, self.params["W3"], self.params["b3"])
        
        scores=out3
        
        
        
        if y is None:
            return scores

        loss, grads = 0, {}
        
        loss, d_start=mse_loss(out3,y)
        loss=loss
        dx3,dw3,db3=affine_backward(d_start,cache3)
        dx2,dw2,db2=affine_sigmoid_backward(dx3,cache2)
        dx1,dw1,db1=affine_sigmoid_backward(dx2,cache1)
        # print(d_start)
        # print(dw3,db3,"layer 3")
        # print(dw2,db2,"layer 2")
        # print(dw1,db1,"layer 1")
        grads["W1"]=dw1+self.params["W1"]*self.reg
        grads["W2"]=dw2+self.params["W2"]*self.reg
        grads["W3"]=dw3+self.params["W3"]*self.reg
        grads["b1"]=db1
        grads["b2"]=db2
        grads["b3"]=db3
        return loss, grads



def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A Solver works on a model object that must conform to the following API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:

      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].

      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].

      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        print('update_rule', self.update_rule)
        if self.update_rule not in globals():
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = globals()[self.update_rule]

        self._reset()


    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config


    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
          'model': self.model,
          'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
          'optim_config': self.optim_config,
          'batch_size': self.batch_size,
          'num_train_samples': self.num_train_samples,
          'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'train_acc_history': self.train_acc_history,
          'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)


    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc


    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            # if first_it or last_it or epoch_end:
            #     train_acc = self.check_accuracy(self.X_train, self.y_train,
            #         num_samples=self.num_train_samples)
            #     val_acc = self.check_accuracy(self.X_val, self.y_val,
            #         num_samples=self.num_val_samples)
            #     self.train_acc_history.append(train_acc)
            #     self.val_acc_history.append(val_acc)
            #     self._save_checkpoint()

            #     if self.verbose:
            #         print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
            #                self.epoch, self.num_epochs, train_acc, val_acc))

            #     # Keep track of the best model
            #     if val_acc > self.best_val_acc:
            #         self.best_val_acc = val_acc
            #         self.best_params = {}
            #         for k, v in self.model.params.items():
            #             self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
        

# ThreeLayer.setWeight()
x=np.array([1,1])[None,:]
y=np.array([1])[None,:]
# ThreeLayer.loss(X=x,y=y)

X_train=np.random.uniform(size=(873,5))
X_val=np.random.uniform(size=(100,5))
y_train=np.random.uniform(size=(873,1))
y_val=np.random.uniform(size=(100,1))
ThreeLayer=ThreeLayerNet(input_dim=(5),hidden_dim=10,outputDim=1)
model = ThreeLayer

data = {'X_train': X_train, 
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val}

##############################################################################
# TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
# 50% accuracy on the validation set.                                        #
##############################################################################


solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=20,
                    print_every=100)
solver.train()