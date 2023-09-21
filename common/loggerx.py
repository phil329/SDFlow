import torch
import os.path as osp
import os
from torchvision.transforms.functional import to_pil_image
import torch.distributed as dist
import inspect
import time
import shutil
import wandb

from .ops import AverageMeter, load_network, reduce_tensor


def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerXBase(object):

    def __init__(self, save_root, print_freq=1,save_history=False):
        self.save_root = save_root
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.dist = dist.is_initialized()
        self.rank = dist.get_rank() if self.dist else 0
        self.world_size = dist.get_world_size() if self.dist else 1
        self.print_freq = print_freq
        self.metrics = {}
        self.save_history = save_history
        if self.save_history:
            self.history = {}

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            if self.rank == 0:
                print('save step {} checkpoint at rank {}...'.format(epoch, self.rank))
                '''d_optim-0030000, discriminator-0030000, g_optim-0030000, generator-0030000'''
                torch.save(module.state_dict(),
                           osp.join(self.models_save_dir, '{}-{}'.format(module_name, str(epoch).zfill(7))))
        if self.dist:
            dist.barrier()

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(
                load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, str(epoch).zfill(7)))))

    def msg_str(self, stats, step=0):
        output_str = '[{}] {:05d}, {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step, str(stats))
        if self.rank == 0:
            print(output_str)

    def msg_internal(self, stats, step, precision=7, print_freq=1):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)
        var_names = []
        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            var_names.append(var_name)
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                if self.dist:
                    var = reduce_tensor(var, self.world_size)
                var = var.item()
            if var_name not in self.metrics:
                self.metrics[var_name] = AverageMeter()
                if self.save_history:
                    self.history[var_name] = []

            self.metrics[var_name].update(var)
            if self.save_history:
                self.history[var_name].append((var))

        output_dict = {}
        if (self.rank == 0) and (step % print_freq == 0):
            for var_name in var_names:
                var = self.metrics[var_name].avg
                output_dict[var_name] = var
                f = '{} {:2.%sf}, ' % precision
                output_str += f.format(var_name, var)
                self.metrics[var_name].reset()
            print(output_str)
        self.msg_handler(output_dict, step)

    def msg(self, stats, step, precision=7):
        self.msg_internal(stats=stats, step=step, precision=precision, print_freq=self.print_freq)

    def msg_metric(self, stats, step, precision=7):
        self.msg_internal(stats=stats, step=step, precision=precision, print_freq=1)

    def msg_handler(self, output_dict, step):
        pass

    def save_image(self, grid_img, n_iter, sample_type):
        if isinstance(grid_img, torch.Tensor):
            grid_img = to_pil_image(grid_img.cpu())
        grid_img.save(osp.join(self.images_save_dir,
                               '{}_{}_{}.png'.format(n_iter, self.rank, sample_type)))

class WANDBLoggerX(LoggerXBase):

    def __init__(self, save_root, print_freq=1, **kwargs):
        '''
        export WANDB_RUN_ID=xxx for resuming
        :param save_root:
        :param print_freq:
        :param kwargs:
        '''
        super().__init__(save_root, print_freq)
        if self.rank == 0:
            wandb.init(dir=save_root, settings=wandb.Settings(_disable_stats=True), **kwargs)

    def msg_handler(self, output_dict, step):
        if self.rank == 0:
            wandb.log(output_dict, step)

    def save_image(self, grid_img, n_iter, sample_type):
        if self.rank != 0:
            return
        if isinstance(grid_img, torch.Tensor):
            grid_img = to_pil_image(grid_img.cpu())
        wandb.log({sample_type: wandb.Image(grid_img, caption=f"{n_iter}_{self.rank}")}, n_iter)
        super().save_image(grid_img, n_iter, sample_type)