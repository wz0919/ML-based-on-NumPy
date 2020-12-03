"""
Minimal character-level LSTM model.
Modified from min-char-rnn.
"""
import numpy as np

# data I/O
data = open('data.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the LSTM for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(4*hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(4*hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((4*hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def lossFun(inputs, targets, prev_c, prev_h):
  """
  inputs,targets are both list of integers.
  prev_h is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, cs, ys, ps, a_s = {}, {}, {}, {}, {}, {}
  hs[-1] = np.copy(prev_h)
  cs[-1] = np.copy(prev_c)
  H = hs[-1].shape[0]
  loss = 0
  # forward pass

  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    a = Wxh @ xs[t] + Whh @ hs[t-1] + bh
    a[:3*H,:] = sigmoid(a[:3*H,:])  #input & forget & output gate
    a[3*H:,:] = np.tanh(a[3*H:,:])  #gate gate
    a_s[t] = a
    i, f, o, g = a[:H,:], a[H:2*H,:], a[2*H:3*H,:], a[3*H:4*H,:]
    cs[t] = f * cs[t-1] + i * g
    hs[t] = o * np.tanh(cs[t])
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dnext_h = np.zeros_like(hs[0])
  dnext_c = np.zeros_like(cs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1
    dnext_h += Why.T @ dy
    dWhy += np.outer(dy, hs[t])
    dby += dy
    i, f, o, g = a_s[t][:H,:], a_s[t][H:2*H,:], a_s[t][2*H:3*H,:], a_s[t][3*H:4*H,:]
    dtanh_nextc = dnext_h * o
    dnext_c += (1-np.tanh(cs[t])**2) * dtanh_nextc
    # dprev_c = dnext_c * f

    dai = dnext_c * g * i*(1-i)
    daf = dnext_c * cs[t-1] * f * (1-f)
    dao = dnext_h * np.tanh(cs[t]) * o * (1-o)
    dag = dnext_c * i * (1 - g**2)
    da = np.vstack((dai, daf, dao, dag))

    dnext_c = dnext_c * f
    dnext_h = Whh.T @ da
    dWhh += da @ hs[t-1].T
    dWxh += da @ xs[t].T
    dbh += da.sum(1, keepdims = True)
  # for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
  #   np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, cs[len(inputs)-1], hs[len(inputs)-1]

def sample(c, h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  H = h.shape[0]
  for t in range(n):
    a = Wxh @ x + Whh @ h + bh
    a[:3*H,:] = sigmoid(a[:3*H,:])  #input & forget & output gate
    a[3*H:,:] = np.tanh(a[3*H:,:])  #gate gate
    i, f, o, g = a[:H,:], a[H:2*H,:], a[2*H:3*H,:], a[3*H:4*H,:]
    c = f * c + i * g
    h = o * np.tanh(c)
    y = np.dot(Why, h) + by # unnormalized log probabilities for next chars
    p = np.exp(y) / np.sum(np.exp(y)) # probabilities for next chars
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
config = []
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
mparam = [mWxh, mWhh, mWhy, mbh, mby].copy()
vparam = mparam.copy()
eps, beta1, beta2 = 1e-8, 0.9, 0.999
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0:
    prev_c = np.zeros((hidden_size,1))
    prev_h = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

   # sample from the model now and then
  if n % 100 == 0:
      sample_ix = sample(prev_c ,prev_h, inputs[0], 200)
      # print(np.linalg.norm(prev_c),np.linalg.norm(prev_h))
      txt = ''.join(ix_to_char[ix] for ix in sample_ix)
      print ('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, prev_c, prev_h = lossFun(inputs, targets, prev_c, prev_h)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  i = 0
  # perform parameter update with Adam
  for param, dparam, m, v in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby],
                                mparam,
                                vparam):
    
    learning_rate = 1e-3 if loss > 5 else 1e-4 if loss > 1 and loss <= 5 else  1e-5 
    m = beta1*m + (1-beta1)*dparam
    mt = m / (1-beta1**(n+1))
    v = beta2*v + (1-beta2)*(dparam**2)
    vt = v / (1-beta2**(n+1))
    param += - learning_rate * mt / (np.sqrt(vt) + eps)
    mparam[i] = m
    vparam[i] = v
    i = i + 1
    
  p += seq_length # move data pointer
  n += 1 # iteration counter 