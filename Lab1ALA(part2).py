import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def read_file(file):
    with open(file, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, n_edges = map(int, f.readline().strip().split())
        verts = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]
        faces = [list(map(int, f.readline().strip().split()[1:])) for _ in range(n_faces)]
    return np.array(verts), faces

vertices, faces = read_file("stairs_0125.off")
print(vertices.shape)

vertices = vertices - vertices.mean(axis=0)

def plot3d(V, title, M=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection([V[f] for f in faces], alpha=0.3, edgecolor='k', linewidths=0.2)
    ax.add_collection3d(mesh)
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, c='r')

    r = np.abs(V).max()
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X");
    ax.set_ylabel("Y");
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.show()

    if M is not None:
        print(title)
        print(np.round(M, 3))


"task3"

def rotate_xy(X, angle):
    X = X.copy()
    cos = np.cos(angle)
    sin = np.sin(angle)
    M = np.array([[cos, -sin, 0],
                  [sin, cos, 0],
                  [0, 0, 1]])
    return (M @ X.T).T, M

def rotate_xz(X, angle):
    X = X.copy()
    cos = np.cos(angle)
    sin = np.sin(angle)
    M = np.array([[ cos, 0, -sin],
                  [ 0, 1, 0],
                  [ sin, 0, cos]])
    return (M @ X.T).T, M

def rotate_yz(X, angle):
    X = X.copy()
    cos = np.cos(angle)
    sin = np.sin(angle)
    M = np.array([[1, 0, 0],
                  [0, cos, -sin],
                  [0, sin, cos]])
    return (M @ X.T).T, M

plot3d(vertices, "From file")

V1, M1 = rotate_xy(vertices, np.radians(45))
plot3d(V1, "Rotation (45) in xy-plane (about z)", M1)

V2, M2 = rotate_yz(vertices, np.radians(45))
plot3d(V2, "Rotation (45) in yz-plane (about x)", M2)

V3, M3 = rotate_xz(vertices, np.radians(45))
plot3d(V3, "Rotation (45) in xz-plane (about y)", M3)

"task4"

XY = rotate_xy(vertices, np.radians(40))[1]
XZ = rotate_xz(vertices, np.radians(20))[1]
YZ = rotate_yz(vertices, np.radians(30))[1]

M_combination = XY @ XZ @ YZ
V_combination = (M_combination @ vertices.T).T
plot3d(V_combination, "Combination  matrix of YZ->XZ->XY ", M_combination)