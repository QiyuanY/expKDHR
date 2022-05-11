def lr_poly(base_lr, _iter, max_iter, _power):
    return base_lr * ((1 - float(_iter) / max_iter) ** (_power))


def adjust_learning_rate(optimozer, i_iter):


# lr = lr_poly(lr, i_iter, num_steps, power)

SEREARSH_PARA = {
    'learning_rate': 0.05,
    'rec': 4e-3,
    'dev_ratio': 0.2,
    'test_ratio': 0.2,
    'ssl_temp': 0.05,
    'ssl_reg': 1e-6,
    'ssl_alpha': 1.5,
    'ncl_k': 20,
    'ncl_k_1': 9,
    'ncl_proto_reg': 8e-8
}

params = {
    'epoch': 200,
    'batch_size': 8192,
    'dropout': 0.0,
    **SEREARSH_PARA
}

