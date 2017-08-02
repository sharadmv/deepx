from .. import T

def kl_divergence(p, q):
    p_param, q_param = p.get_parameters('natural'), q.get_parameters('natural')
    p_dim, q_dim = len(T.get_shape(p_param)), len(T.get_shape(q_param))
    p_stats = p.expected_sufficient_statistics()
    p_log_z, q_log_z = p.log_z(), q.log_z()
    if p_dim == 1:
        p_param = p_param[None]
        p_stats = p_stats[None]
        p_log_z = p_log_z[None]
        p_dim = 2
    while p_dim > q_dim:
        q_param = q_param[None]
        q_log_z = q_log_z[None]
        q_dim = len(T.get_shape(q_param))
    while q_dim > p_dim:
        p_param = p_param[None]
        p_stats = p_stats[None]
        p_log_z = p_log_z[None]
        p_dim = len(T.get_shape(p_param))
    return T.sum((p_param - q_param) * p_stats, axis=list(range(1, p_dim))) - p_log_z + q_log_z
