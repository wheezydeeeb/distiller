import sys
import torch
import math
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import DataParallel
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm

from optimizer import get_optimizer, get_scheduler

# Custom Focal Loss Function
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# EMA Optimizer for distillation framework
class WeightEMA(optim.Optimizer):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def init_progress_bar(train_loader):
    batch_size = train_loader.batch_size
    bar_format = "{desc}{percentage:3.0f}%"
    # bar_format += "|{bar}|"
    bar_format += " {n_fmt}/{total_fmt} [{elapsed} < {remaining}]"
    bar_format += "{postfix}"
    # if stderr has no tty disable the progress bar
    disable = not sys.stderr.isatty()
    t = tqdm(total=len(train_loader) * batch_size,
             bar_format=bar_format, disable=disable)
    if disable:
        # a trick to allow execution in environments where stderr is redirected
        t._time = lambda: 0.0
    return t

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class Trainer():
    def __init__(self, net, config):

        """ ----------------------------
        LOSS FUNCTION INITIALIZATION
        -----------------------------"""
        # self.loss_fun = nn.CrossEntropyLoss()
        self.loss_fun = FocalLoss(gamma=2)

        self.net = net
        self.device = config["device"]
        self.name = config["test_name"]

        # Retrieve preconfigured optimizers and schedulers for all runs
        optim = config["optim"]
        sched = config["sched"]
        self.optim_cls, self.optim_args = get_optimizer(optim, config)
        self.sched_cls, self.sched_args = get_scheduler(sched, config)
        self.optimizer = self.optim_cls(net.parameters(), **self.optim_args)
        self.scheduler = self.sched_cls(self.optimizer, **self.sched_args)

        self.train_loader = config["train_loader"]
        self.test_loader = config["test_loader"]
        self.batch_size = self.train_loader.batch_size
        self.config = config

        # tqdm bar
        self.t_bar = None
        folder = config["results_dir"]
        self.best_model_file = folder.joinpath(f"{self.name}_best.pth.tar")
        acc_file_name = folder.joinpath(f"{self.name}_train.csv")
        self.acc_file = acc_file_name.open("w+")
        self.acc_file.write("Train Acc,Val Acc,Train Loss,Val Loss\n")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_loss_fun(self, loss_fun):
        self.loss_fun = loss_fun

    def calculate_loss(self, data, target):
        raise NotImplementedError(
            "calculate_loss should be implemented by subclass!")

    """First and Second Step Implementation for SAM"""
    def calculate_loss_first(self, data, target):
        raise NotImplementedError(
            "calculate_loss should be implemented by subclass!")

    def calculate_loss_second(self, data, target):
        raise NotImplementedError(
            "calculate_loss should be implemented by subclass!")

    def train_single_epoch(self, t_bar):
        self.net.train()
        total_correct = 0.0
        total_loss = 0.0
        len_train_set = len(self.train_loader.dataset)
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device).long()
            
            self.optimizer.zero_grad()
            enable_running_stats(self.net)
            y_hat, loss = self.calculate_loss_first(x, y)
            # y_hat, loss = self.calculate_loss(x, y)
            self.optimizer.zero_grad()
            disable_running_stats(self.net)
            y_hat_adv, loss_adv = self.calculate_loss_second(x, y)

            # Metric tracking boilerplate
            total_loss += loss
            curr_loss = (total_loss / float(batch_idx))

            y_hat = y_hat.data.cpu().numpy()
            y_hat = np.argmax(y_hat, axis=1)
            y = y.data.cpu().numpy()
            
            curr_acc = 100 * np.mean((y_hat == y).astype(int))

            t_bar.update(self.batch_size)
            t_bar.set_postfix_str(f"Acc {curr_acc:.3f}% Loss {curr_loss:.3f}")
        # total_acc = float(total_correct / len_train_set)
        total_acc = curr_acc
        total_loss = total_loss / float(batch_idx)
        return total_acc, total_loss

    def train(self):
        epochs = self.config["epochs"]

        best_acc = 0
        t_bar = init_progress_bar(self.train_loader)
        for epoch in range(epochs):
            # update progress bar
            t_bar.reset()
            t_bar.set_description(f"Epoch {epoch}")
            # perform training
            train_acc, train_loss = self.train_single_epoch(t_bar)
            # validate the output and save if it is the best so far
            val_acc, val_loss = self.validate(epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                self.save(epoch, name=self.best_model_file)
            # update the scheduler
            if self.scheduler:
                self.scheduler.step()
            self.acc_file.write(f"{train_acc},{val_acc},{train_loss},{val_loss}\n")
        tqdm.clear(t_bar)
        t_bar.close()
        self.acc_file.close()
        return best_acc

    def validate(self, epoch=0):
        self.net.eval()
        acc = 0.0
        with torch.no_grad():
            correct = 0
            acc = 0
            loss = 0
            for idx, (images, labels) in enumerate(self.test_loader):
                # images = images.repeat(1, 3, 1, 1).to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                output = self.net(images, labels)
                # Standard Learning Loss ( Classification Loss)
                loss += self.loss_fun(output, labels)
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

            acc = float(correct) / len(self.test_loader.dataset)
            loss = loss / float(idx)
            print(f"\nEpoch {epoch}: Validation set: Average loss: {loss:.4f},"
                  f" Accuracy: {correct}/{len(self.test_loader.dataset)} "
                  f"({acc * 100.0:.3f}%)")
        return acc, loss

    def save(self, epoch, name):
        torch.save({"model_state_dict": self.net.state_dict(), }, name)

"""----------BASE TRAINING CLASS FOR TEACHER----------"""
class BaseTrainer(Trainer):

    def calculate_loss(self, data, target):
        # Standard Learning Loss ( Classification Loss)
        output = self.net(data, target)
        loss = self.loss_fun(output, target)
        loss.backward()
        self.optimizer.step()
        return output, loss

    def calculate_loss_first(self, data, target):
        output = self.net(data, target)
        loss = self.loss_fun(output, target)
        loss.mean().backward()
        self.optimizer.first_step()
        return output, loss

    def calculate_loss_second(self, data, target):
        output = self.net(data, target)
        loss = self.loss_fun(output, target)
        loss.mean().backward()
        self.optimizer.second_step()
        return output, loss

"""----------BASE DISILLATION CLASS----------"""
class KDTrainer(Trainer):
    def __init__(self, s_net, t_net, config):
        super(KDTrainer, self).__init__(s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        self.kd_fun = nn.KLDivLoss(size_average=False)

    def kd_loss(self, out_s, out_t, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        # Standard Learning Loss ( Classification Loss)
        loss = self.loss_fun(out_s, target)
        # Knowledge Distillation Loss
        batch_size = target.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        return loss

    def calculate_loss(self, data, target):
        out_s = self.s_net(data)
        out_t = self.t_net(data)
        loss = self.kd_loss(out_s, out_t, target)
        loss.backward()
        self.optimizer.step()
        return out_s, loss


class TripletTrainer(KDTrainer):
    def __init__(self, s_net, t_net, config):
        super(TripletTrainer, self).__init__(s_net, t_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        self.triplet = F.cosine_embedding_loss

    def kd_loss(self, out_s, out_t, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        # Standard Learning Loss ( Classification Loss)
        # loss = self.loss_fun(out_s, target)
        # Knowledge Distillation Loss
        batch_size = target.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        # pred_s = out_s.data.max(1, keepdim=True)[1]
        # pred_t = out_t.data.max(1, keepdim=True)[1]
        y = torch.ones(target.shape[0]).cuda()
        loss = self.triplet(out_s, out_t, y)
        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        return loss

    def calculate_loss(self, data, target):
        out_s = self.s_net(data)
        out_t = self.t_net(data)
        loss = self.kd_loss(out_s, out_t, target)
        loss.backward()
        self.optimizer.step()
        return out_s, loss


class MultiTrainer(KDTrainer):
    def __init__(self, s_net, t_nets, config):
        super(MultiTrainer, self).__init__(s_net, s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_nets = t_nets
        # EMA Optimizer Definition
        self.ema_optimizer = WeightEMA(model=self.s_net, ema_model=self.s_net, lr=config["learning_rate"])

    def kd_loss(self, out_s, out_t, target):
        T = self.config["T_student"]
        # Knowledge Distillation Loss
        batch_size = target.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        return loss_kd

    def calculate_loss(self, data, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        out_s = self.s_net(data, target)
        # Standard Learning Loss (Classification Loss)
        loss = self.loss_fun(out_s, target)
        # Average Knowledge Distillation Loss
        # loss_kd = 0.0
        # for t_net in self.t_nets:
        #     out_t = t_net(data)
        #     loss_kd += self.kd_loss(out_s, out_t, target)
        # loss_kd /= len(self.t_nets)

        # Maximum Voting Knowledge Distillation Loss
        loss_kd_list = [self.kd_loss(out_s, t_net(data, target), target) for t_net in self.t_nets]
        loss_kd = max(loss_kd_list)

        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        loss.backward()
        self.optimizer.step()
        return out_s, loss

    def calculate_loss_first(self, data, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        out_s = self.s_net(data, target)
        loss = self.loss_fun(out_s, target)
        loss_kd_list = [self.kd_loss(out_s, t_net(data, target), target) for t_net in self.t_nets]
        loss_kd = max(loss_kd_list)
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        loss.mean().backward()
        self.optimizer.first_step()
        # Stepping the ema_optimizer
        self.ema_optimizer.step()
        return out_s, loss

    def calculate_loss_second(self, data, target):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        out_s = self.s_net(data, target)
        loss = self.loss_fun(out_s, target)
        loss_kd_list = [self.kd_loss(out_s, t_net(data, target), target) for t_net in self.t_nets]
        loss_kd = max(loss_kd_list)
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        loss.mean().backward()
        self.optimizer.second_step()
        # Stepping the ema_optimizer
        self.ema_optimizer.step()
        return out_s, loss


class BlindTrainer(KDTrainer):
    def __init__(self, s_net, t_net, config):
        super(BlindTrainer, self).__init__(s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net

    def calculate_loss(self, data):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        out_s = self.s_net(data)

        # Knowledge Distillation Loss
        out_t = self.t_net(data)
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        batch_size = s_max.shape[0]
        loss_kd = F.kl_div(s_max, t_max, size_average=False) / batch_size
        loss = lambda_ * T * T * loss_kd
        loss.backward()
        self.optimizer.step()
        return out_s, loss

    def train_single_epoch(self, t_bar):
        self.net.train()
        total_loss = 0
        iters = int(len(self.train_loader.dataset) / self.batch_size)
        for batch_idx in range(iters):
            data = torch.randn((self.batch_size, 3, 32, 32)).to(self.device)
            self.optimizer.zero_grad()
            loss = self.calculate_loss(data)
            total_loss += loss
            t_bar.update(self.batch_size)
            loss_avg = total_loss / batch_idx
            t_bar.set_postfix_str(f"Loss {loss_avg:.6f}")
        return total_loss / len(self.train_loader.dataset)
