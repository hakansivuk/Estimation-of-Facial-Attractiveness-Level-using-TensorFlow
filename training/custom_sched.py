def set_lr(epoch, optimizer):
    if epoch > 30:
        optimizer.lr = 0.0001
    if epoch > 80:
        optimizer.lr = 0.00001