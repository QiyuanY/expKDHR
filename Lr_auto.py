
config

def lr_poly(base_lr, _iter, max_iter, _power):
    return base_lr * ((1 - float(_iter) / max_iter) ** (_power))

def adjust_learning_rate(optimozer, i_iter):
    lr = lr_poly(lr, i_iter, num_steps, power)