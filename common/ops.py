import collections.abc as container_abcs
import warnings
from typing import Union

from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn

def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def no_distributed(func):
    def wrapper(*args, **kw):
        if dist.is_initialized():
            if dist.get_rank() == 0:
                results = func(*args, **kw)
            else:
                results = None
            dist.barrier()
        else:
            results = func(*args, **kw)
        return results

    return wrapper


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def turn_on_spectral_norm(module):
    module_output = module
    # if isinstance(module, torch.nn.Conv2d):
    #     if module.out_channels != 1 and module.in_channels > 4:
    #         module_output = nn.utils.spectral_norm(module)
    # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #     module_output = nn.utils.spectral_norm(module)
    for name, child in module.named_children():
        module_output.add_module(name, turn_on_spectral_norm(child))
    del module
    return module_output


def apply_weight_decay(*modules, weight_decay_factor=0., wo_bn=True):
    '''
    Works only for SGD with momentum
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/5
    Apply weight decay to pytorch model without BN;
    In pytorch:
        if group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])
    p is the param;
    :param modules:
    :param weight_decay_factor:
    :return:
    '''
    for module in modules:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Module.modules
        for m in module.modules():
            if hasattr(m, 'weight'):
                if wo_bn and isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    continue
                if m.weight.grad is None:
                    continue
                m.weight.grad.add_(m.weight, alpha=weight_decay_factor)


@torch.no_grad()
def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def get_dex_age(pred):
    pred = F.softmax(pred, dim=1)
    value = torch.sum(pred * torch.arange(pred.size(1)).to(pred.device), dim=1)
    return value


def normalize(input, mean, std):
    mean = torch.Tensor(mean).to(input.device)
    std = torch.Tensor(std).to(input.device)
    return input.sub(mean[None, :, None, None]).div(std[None, :, None, None])


# from https://github.com/NVlabs/DG-Net/blob/0abf564a853ea6ec3f38ab71a4a69f7f23b19d24/networks.py#L155
# regularize real grad
def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def pfa_encoding(source, target, age_group):
    source, target = source.long(), target.long()
    code = torch.zeros((source.size(0), age_group - 1, 1, 1, 1)).to(source)
    for i in range(source.size(0)):
        code[i, source[i]: target[i], ...] = 1
    return code


def get_act_layer(act_layer, **kwargs):
    if act_layer == 'none':
        return nn.Identity()
    elif act_layer == 'relu':
        return nn.ReLU(inplace=True)
    elif act_layer == 'lrelu':
        return nn.LeakyReLU(inplace=True, negative_slope=0.2)
    else:
        raise NotImplementedError


def get_norm_layer(norm_layer, module, **kwargs):
    if norm_layer == 'none':
        return module
    elif norm_layer == 'bn':
        return nn.Sequential(
            module,
            nn.BatchNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'in':
        return nn.Sequential(
            module,
            nn.InstanceNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'sn':
        return nn.utils.spectral_norm(module, **kwargs)
    elif norm_layer == 'pn':
        from common.nn.modules import PixelNormLayer
        return nn.Sequential(
            module,
            PixelNormLayer(**kwargs)
        )
    else:
        return NotImplementedError


def compute_grad_norm(*modules):
    import itertools
    parameters = list(itertools.chain.from_iterable([list(m.parameters()) for m in modules]))
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = 2.0
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
                            norm_type)
    return total_norm


def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda(non_blocking=True)
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_to_cuda(d) for d in data))
    else:
        return data


def desired_labels(labels, num_class):
    mask = torch.ones((labels.size(0), num_class))
    mask[torch.arange(labels.size(0)), labels] = 0
    _labels = torch.multinomial(mask, 1, replacement=True)
    return _labels.squeeze().to(labels)


def age2group(age, age_group):
    if isinstance(age, np.ndarray):
        groups = np.zeros_like(age)
    else:
        groups = torch.zeros_like(age).to(age.device)
    if age_group == 2:
        section = [40, ]
    elif age_group == 4:
        section = [30, 40, 50]
    elif age_group == 5:
        section = [20, 30, 40, 50]
    elif age_group == 7:
        section = [10, 20, 30, 40, 50, 60]
    elif age_group == 10:
        section = [2, 6, 9, 14, 19, 29, 39, 49, 69]
    else:
        section = [40,]
        # raise NotImplementedError
    for i, thresh in enumerate(section, start=1):   #  下标从 1 开始返回序列 idx 和 element
        groups[age > thresh] = i
    return groups


def label2onehot(labels, num_class):
    code = torch.eye(num_class)[labels.long().squeeze()]
    if len(code.size()) > 1:
        return code
    return code.unsqueeze(0).to(labels)


def label2map(labels, num_class, size):
    return onehot2map(label2onehot(labels, num_class), size).to(labels)


def multilabel2map(labels, num_classes, size):
    map_list = []
    for i in range(labels.size(1)):
        label_map = label2map(labels[:, i], num_classes[i], size)
        map_list.append(label_map)
    return torch.cat(map_list, dim=1).to(labels)


def onehot2map(onehots, size):
    return onehots.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, size, size)


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is not None:
        rt /= world_size
    return rt


def convert_to_ddp(modules: Union[list, nn.Module],
                   optimizer: torch.optim.Optimizer = None,
                   opt_level: int = 0,
                   **kwargs):
    if isinstance(modules, list):
        modules = [x.cuda() for x in modules]
    else:
        modules = modules.cuda()
    try:
        import apex
        from apex import amp
        use_apex = True
    except:
        use_apex = False
    if optimizer is None:
        use_apex = False
    if use_apex:
        if optimizer is not None:
            modules, optimizer = amp.initialize(modules, optimizer, opt_level="O{}".format(opt_level), verbosity=1)
        if isinstance(modules, list):
            modules = [apex.parallel.DistributedDataParallel(x, delay_allreduce=True) for x in modules]
        else:
            modules = apex.parallel.DistributedDataParallel(modules, delay_allreduce=True)
    else:
        if dist.is_initialized():
            rank = dist.get_rank()
            if isinstance(modules, list):
                modules = [torch.nn.parallel.DistributedDataParallel(x,
                                                                     device_ids=[rank, ],
                                                                     output_device=rank, **kwargs) for
                           x in modules]
            else:
                modules = torch.nn.parallel.DistributedDataParallel(modules,
                                                                    device_ids=[rank, ],
                                                                    output_device=rank, **kwargs)

        else:
            modules = torch.nn.DataParallel(modules)

    if optimizer is not None:
        return modules, optimizer
    else:
        return modules


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`  #du: namekey = k[7:]
        new_state_dict[namekey] = v
    return new_state_dict


# from common.nn.insightface import iresnet50
# preprocess for insightface image input
def process_for_id(x,img_size=1024):
    scale = lambda x: int(x * img_size / 256)
    h, x1, x2 = scale(188), scale(35), scale(32)
    x = x[:, :, x1:x1 + h, x2:x2 + h]
    x = F.interpolate(x, size=112, mode='bilinear', align_corners=True)
    # x = x * 128 / 255   # [0,1] --> [-1,1]
    return x

# initial all parameters to zero
def init_weights(m):
    if (type(m) == nn.Linear) or (type(m) == nn.LayerNorm) or(type(m) == nn.BatchNorm2d):
        init.zeros_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif type(m) == nn.Conv2d:
        init.zeros_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif type(m) == nn.MultiheadAttention:
        # zeros-initialization  q_weight, k_weight, v_weight, out_weight
        init.zeros_(m.in_proj_weight)
        if m.in_proj_bias is not None:
            init.zeros_(m.in_proj_bias)
        init.zeros_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            init.zeros_(m.out_proj.bias)
