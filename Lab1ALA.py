import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('Bugs_bunny.png')

if img.shape[2] == 4:
    mask = img[:, :, 3] > 0.5
else:
    mask = img[:, :, :3].mean(axis=2) < 0.9

ys, xs = np.nonzero(mask)
step = 3
ys, xs = ys[::step], xs[::step]
colors = img[ys, xs, :3]

B = np.vstack([xs, -ys]).astype(float)
B -= B.mean(axis=1, keepdims=True)
print(B.shape)

def show(X, title, M=None):
    plt.figure(figsize=(5, 5))
    plt.scatter(X[0], X[1], c=colors, s=1)
    plt.axhline(0, color='lightgray')
    plt.axvline(0, color='lightgray')
    plt.axis('equal')
    plt.title(title)
    plt.show()
    if M is not None:
        print(title)
        print(M)


show(B, "Original")

"task1"

def stretch(X,a,b):
    X = X.copy()
    M = np.array([[a, 0],[0, b]])
    return M @ X, M

def shear(X,a,b):
    X = X.copy()
    M = np.array([[1, a], [b, 1]])
    return M @ X, M

def reflection(X,a,b):
    X = X.copy()
    M = np.array([[a**2 - b**2, 2*a*b],
                  [2*a*b, b**2 - a**2]]) / (a**2 + b**2)
    return M @ X, M

def rotation(X,angle):
    X = X.copy()
    M = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return M @ X, M

"demo"
X1, M1 = stretch(B, 1.5, 0.7)
show(X1, "Stretch (1.5, 0.7)", M1)
X2, M2 = shear(B, 0.5, 0)
show(X2, "Shear (0.5, 0)", M2)
X3, M3 = reflection(B, 1, 1)
show(X3, "Reflection (y=x)", M3)
X4, M4 = rotation(B, np.radians(45))
show(X4, "Rotation 45°", M4)

"experiments"
for a,b in [(0.2, 0.2), (-1, 1), (1, 0)]:
    X, M = stretch(B, a, b);  show(X, f"Stretch ({a}, {b})", M)
for a, b in [(0, 0.5), (0.5, 0.5)]:
    X, M = shear(B, a, b);    show(X, f"Shear ({a}, {b})", M)

for a, b in [(1, 0), (2, 2)]:
    X, M = reflection(B, a, b);  show(X, f"Reflection ({a}, {b})", M)

for deg in [90, -45]:
    X, M = rotation(B, np.radians(deg));  show(X, f"Rotation {deg}°", M)

"task2"
S = stretch(B, 1.5, 0.7)[1]
SH = shear(B, 0.5, 1)[1]
R = rotation(B, np.radians(45))[1]

M1 = R @ SH @ S
M2 = SH @ S @ R
M3 = S @ R @ SH

show(M1 @ B, "Stretch → Shear → Rotation", M1)
show(M2 @ B, "Rotation → Stretch → Shear", M2)
show(M3 @ B, "Shear → Rotation → Stretch", M3)

print("Comparison of matrices")
print("Is M1 and M2 is equal?", np.allclose(M1, M2))
print("Is M1 and M3 is equal?",np.allclose(M1, M3))
print("Is M2 and M3 is equal?",np.allclose(M2, M3))
