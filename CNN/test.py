import numpy as np
from matplotlib import pyplot as plt
from cnn import *

'''
Overfitting randomly generated 100 noise 
and 100 labels to test if cnn works
'''

cnn = miniCNN()

x = np.random.randn(100,3,32,32)
y = np.random.randint(0,9,100)
losses = []
accuracies = []
accuracy = 0

while accuracy < 1:
    out, cache = cnn.forward(x)
    loss, dout = cross_entropy(out, y)
    accuracy = np.mean(out.argmax(1)==y)
    print('loss: {:.4}, accuracy: {}%'.format(loss, int(accuracy*100)))
    losses.append(loss)
    accuracies.append(accuracy)
    dout, grad = cnn.backward(dout, cache)
    cnn.update(grad)

plt.subplot(1, 2, 1)
plt.title('Accuracy')
plt.plot(accuracies)
plt.xlabel('iteration')
plt.ylabel('rate')

plt.subplot(1, 2, 2)
plt.title('Loss')
plt.plot(losses)
plt.xlabel('iteration')
plt.ylabel('loss')

plt.tight_layout()
plt.savefig('result.png')
plt.show()