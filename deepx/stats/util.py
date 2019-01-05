from .. import T

def coerce_param(p_param, q_param):
    if isinstance(p_param, list):
        return list(map(list, zip(*[coerce_param(a, b) for a, b in zip(p_param, q_param)])))
    dim_p = len(T.get_shape(p_param))
    dim_q = len(T.get_shape(q_param))
    while dim_p > dim_q:
        q_param = q_param[None]
        dim_q += 1
    while dim_q > dim_p:
        p_param = p_param[None]
        dim_p += 1
    return p_param, q_param

def kl_divergence(p, q):
    param_dim = p.get_param_dim()
    dist = p.__class__
    p_param, q_param = coerce_param(p.get_parameters('natural'), q.get_parameters('natural'))
    p, q = dist(p_param, 'natural'), dist(q_param, 'natural')
    p_stats = p.expected_sufficient_statistics()
    p_log_z = p.log_z()
    q_log_z = q.log_z()
    if isinstance(p_param, list):
        return sum([T.sum((a - b) * c, axis=list(range(-p_d, 0))) for a, b, c, p_d in zip(p_param, q_param, p_stats, param_dim)]) - p_log_z + q_log_z
    return T.sum((p_param - q_param) * p_stats, axis=list(range(-param_dim, 0))) - p_log_z + q_log_z

def pack(parameters):
    A, b, kappa, nu = parameters
    leading_dim, D = T.shape(b)[:-1], T.shape(b)[-1]
    z1 = T.zeros(T.concat([leading_dim, [D, 1]], 0), dtype=T.dtype(A))
    z2 = T.zeros(T.concat([leading_dim, [1, 1]], 0), dtype=T.dtype(A))
    b = b[...,None]
    kappa, nu = (
        T.reshape(kappa,
                    T.concat([leading_dim, [1, 1]], 0)),
        T.reshape(nu,
                    T.concat([leading_dim, [1, 1]], 0))
    )
    return vs(( hs(( A,     b,      z1 )),
                hs(( t(z1), kappa,  z2 )),
                hs(( t(z1), z2,     nu ))))

def unpack(A):
    D = T.shape(A)[-1] - 2
    return [A[...,:D, :D], A[...,:D,D], A[...,D,D], A[...,D+1,D+1]]

hs = lambda x: T.concat(x, -1)
vs = lambda x: T.concat(x, -2)
t  = lambda x: T.matrix_transpose(x)
