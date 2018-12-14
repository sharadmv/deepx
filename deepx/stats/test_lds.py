import numpy as np

from deepx import stats, T

N, H, ds, da = 1, 2, 4, 2

# random rotation for state-state transition
A = np.zeros([H-1, ds, ds])
for t in range(H-1):
    theta = 0.5 * np.pi * np.random.rand()
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = np.zeros((ds, ds))
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(ds, ds))[0]
    A[t] = q.dot(out).dot(q.T)
A = T.constant(A, dtype=T.floatx())

B = T.constant(0.1 * np.random.randn(H-1, ds, da), dtype=T.floatx())
Q = T.matrix_diag(np.random.uniform(low=0.9, high=1.1, size=[H-1, ds]).astype(np.float32))

prior = stats.Gaussian([T.eye(ds), T.zeros(ds)])
p_S = stats.Gaussian([T.eye(ds, batch_shape=[N, H]), T.constant(np.random.randn(N, H, ds), dtype=T.floatx())])
potentials = stats.Gaussian.unpack(p_S.get_parameters('natural')) + [p_S.log_z()]
actions = T.constant(np.random.randn(N, H, da), dtype=T.floatx())

lds = stats.LDS(((A, B, Q), prior, potentials, actions))

sess = T.interactive_session()

np.set_printoptions(suppress=True, precision=2, edgeitems=100, linewidth=1e8, threshold=1e8)
