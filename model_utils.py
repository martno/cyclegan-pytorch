import utils


def compute_num_weights(model):
    return sum(utils.product(param.size()) for param in model.parameters())
