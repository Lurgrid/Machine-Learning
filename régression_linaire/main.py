import matplotlib.pyplot as plt
import torch
from random import uniform
import numpy as np

# Modèle Linaire: f(x) = a * x + b
# Paramètre inconnu: a et b
# Fonction de coût: T(a, b) = 1/m * sum (f(x^i) - y^i)^2

# Génération des données d'entrée (superficie)
X = torch.linspace(-100, 100, steps=30)

hide_a = uniform(1, 20)
hide_b = uniform(1, 20)

# Génération des données de sortie (prix)
Y = hide_a + hide_b * X + torch.randn(30) * 100  
# Relation linéaire avec bruit ajouté

rand_a = uniform(1, 20)
rand_b = uniform(1, 20)

print("Value at start: ", float(rand_a), " + ", float(rand_b), " * x")

alpha = 0.0001

precition = 0.01

while True:
    dac = ((rand_b * X + rand_a - Y) / len(X)).sum()
    dbc = ((X * (rand_b * X + rand_a - Y)) / len(X)).sum()
    rand_a -= alpha * dac
    rand_b -= alpha * dbc
    if abs(dac) > precition or abs(dbc) > precition:
        continue
    else:
        break

# Tracé des données générées
plt.scatter(X.numpy(), Y.numpy())

x = np.linspace(-100, 100)
y = rand_a + rand_b * x
plt.plot(x, y, "r")

print("Value find: ", float(rand_a), " + ", float(rand_b), " * x")
print("True value: ", hide_a, " + ", hide_b, " * x")

rmce = float(np.sqrt(1 / len(X) * (((rand_a + rand_b * X) - Y) ** 2).sum()))
moy = Y.sum() / len(Y)
# coefficient de corrélation multiple
r2 = 1 - ((((rand_a + rand_b * X) - Y) ** 2).sum() / ((Y - moy) ** 2).sum())

plt.xlabel('x')
plt.ylabel('f(x) = y')
plt.text(70, 0, "RMCE = {:.2f}".format(rmce), ha='center')
plt.text(70, Y.max() / 2, "R² = {:.3f}".format(r2), ha='center')
plt.text(-70, 0, "{:.2f} * x + {:.2f}".format(rand_b, rand_a), ha='center')
plt.show()

