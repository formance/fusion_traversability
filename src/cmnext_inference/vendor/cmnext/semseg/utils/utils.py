def nchw_to_nlc(x):
    return x.flatten(2).transpose(1, 2).contiguous()
def nlc_to_nchw(x, hw_shape):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], *hw_shape).contiguous()
