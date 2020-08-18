import sys
from collections import OrderedDict

import torch
from torch.nn import functional as F

class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args

class Sequential(torch.nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
        
        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        # i = 0
        for module in self._modules.values():        #3个block,每个18个层
            # print(i)
            input = module(input)
            a=input.shape
            # i += 1
        return input
# OrderedDict([('blocks', ModuleList(
#   (0): Sequential(
#     (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
#     (1): DefaultArgLayer(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
#     (2): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (3): ReLU()
#     (4): DefaultArgLayer(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (5): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (6): ReLU()
#     (7): DefaultArgLayer(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (8): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (9): ReLU()
#     (10): DefaultArgLayer(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (11): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (12): ReLU()
#   )
#   (1): Sequential(
#     (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
#     (1): DefaultArgLayer(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
#     (2): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (3): ReLU()
#     (4): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (5): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (6): ReLU()
#     (7): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (8): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (9): ReLU()
#     (10): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (11): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (12): ReLU()
#     (13): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (14): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (15): ReLU()
#     (16): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (17): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (18): ReLU()
#   )
#   (2): Sequential(
#     (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
#     (1): DefaultArgLayer(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
#     (2): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (3): ReLU()
#     (4): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (5): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (6): ReLU()
#     (7): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (8): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (9): ReLU()
#     (10): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (11): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (12): ReLU()
#     (13): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (14): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (15): ReLU()
#     (16): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (17): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (18): ReLU()
#   )
# )), ('deblocks', ModuleList(
#   (0): Sequential(
#     (0): DefaultArgLayer(64, 128, kernel_size=(4, 4), stride=(4, 4), bias=False)
#     (1): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (2): ReLU()
#   )
#   (1): Sequential(
#     (0): DefaultArgLayer(128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
#     (1): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (2): ReLU()
#   )
#   (2): Sequential(
#     (0): DefaultArgLayer(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (1): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#     (2): ReLU()
#   )
# )), ('conv_cls', Conv2d(384, 200, kernel_size=(1, 1), stride=(1, 1))), ('conv_box', Conv2d(384, 140, kernel_size=(1, 1), stride=(1, 1))), ('conv_dir_cls', Conv2d(384, 40, kernel_size=(1, 1), stride=(1, 1)))])