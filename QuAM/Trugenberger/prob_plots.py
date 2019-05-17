import numpy as np
import matplotlib.pyplot as plt

# perfect distribution with b bits
bits = 62

b_vals = np.arange(1, 10)
dists = np.arange(bits)
for b in b_vals:
    probs = np.cos(np.pi/(2*bits)*dists)**(2*b)
    probs = probs/np.sum(probs)
    plt.plot(dists, probs, label="b = " + str(b))
plt.xlabel("Hamming distance")
plt.ylabel("Probability")
plt.title("Probability distribution auxiliary bit dependence")
plt.legend()
plt.grid()
plt.show()


b = 10
dist_vals = np.arange(5)
dists = np.arange(bits)
for l in dist_vals:
    probs = np.cos(np.pi/(2*bits)*dists)**(2*(b-l))*np.sin(np.pi/(2*bits)*dists)**(2*l)
    probs = probs/np.sum(probs)
    plt.plot(dists, probs, label="l = " + str(l))
plt.xlabel("Hamming distance")
plt.ylabel("Probability")
plt.title("Probability distribution, auxiliary measurement result")
plt.legend()
plt.grid()
plt.show()

b = 50
dist_vals = np.arange(5)
dists = np.arange(bits)
for l in dist_vals:
    probs = np.cos(np.pi/(2*bits)*dists)**(2*(b-l))*np.sin(np.pi/(2*bits)*dists)**(2*l)
    probs = probs/np.sum(probs)
    plt.plot(dists, probs, label="l = " + str(l))

plt.xlabel("Hamming distance")
plt.ylabel("Probability")
plt.title("Probability distribution, auxiliary measurement result, more variables")
plt.legend()
plt.grid()
plt.show()
