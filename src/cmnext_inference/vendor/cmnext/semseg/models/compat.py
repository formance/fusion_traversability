from torch import nn

def build_norm_layer(cfg, num_features, postfix=''):
    # CMNeXt's MSPA blocks use mmcv's default BN configuration only.
    options = dict(cfg)
    if options.pop('type') != 'BN':
        raise ValueError('The inference adapter supports BN only')
    requires_grad = options.pop('requires_grad', True)
    layer = nn.BatchNorm2d(num_features, **options)
    for parameter in layer.parameters():
        parameter.requires_grad_(requires_grad)
    return 'bn' + str(postfix), layer
