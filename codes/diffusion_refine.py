from typing import Tuple, Optional
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import config
from tqdm import tqdm

def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    consts:(time_steps,)
    t:(b,)
    return: (b,1,1,1)
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class DenoiseDiffusion:
    def __init__(self, eps_model: nn.Module, n_steps: int,device):
        """
        * `eps_model`   输入t,xt,输出噪声
        * `n_steps` 最大步数
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model
        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps,device = device)

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta


       
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### q(x_t|x_0),从x_0得到x_t
        x0:(b,c,w,h)
        t:(b,)
        return: [mean:(b,1,1,1)  var:(b,1,1,1)]
        """
        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$
        return: x_t (b,c,w,h)
        """
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor,x0, t: torch.Tensor,condition:torch.tensor):
        """
        从 xt 得到xt-1
        return: x_t-1:(b,c,w,h)
        """

        eps_theta = self.eps_model(xt,x0, t,condition=condition)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5

        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)

        var = gather(self.sigma2, t)

        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps
    
    def p_sample_ddim(self,
                      img_or_shape,
                      x0,
                      condition,
                      simple_var=True,
                      ddim_step=100,
                      eta=1):
        if simple_var:
            eta = 1
        device=config.device
        ts = torch.linspace(self.n_steps, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        net=self.eps_model
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)
        batch_size = x.shape[0]
        net = net.to(device)
        for i in tqdm(range(1, ddim_step + 1),
                      f'DDIM sampling with eta {eta} simple_var {simple_var}'):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            ab_cur = self.alpha_bar[cur_t]
            ab_prev = self.alpha_bar[prev_t] if prev_t >= 0 else 1

            #t_tensor = torch.tensor([cur_t] * batch_size, dtype=torch.long).to(device).unsqueeze(1)
            t_tensor=x.new_full((config.n_samples,),fill_value=cur_t, dtype=torch.long)
            eps = net(x,x0, t_tensor,condition=condition)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur)**0.5 * x
            second_term = ((1 - ab_prev - var)**0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x = first_term + second_term + third_term

        return x
    
    def classifire_p_sample_ddim(self,
                      img_or_shape,
                      x0,
                      condition,
                      simple_var=True,
                      ddim_step=100,
                      eta=1,
                      scale=1.):
        if simple_var:
            eta = 1
        device=config.device
        ts = torch.linspace(self.n_steps, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        net=self.eps_model
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)
        batch_size = x.shape[0]
        net = net.to(device)
        for i in tqdm(range(1, ddim_step + 1),
                      f'DDIM sampling with eta {eta} simple_var {simple_var}'):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            ab_cur = self.alpha_bar[cur_t]
            ab_prev = self.alpha_bar[prev_t] if prev_t >= 0 else 1

            #t_tensor = torch.tensor([cur_t] * batch_size, dtype=torch.long).to(device).unsqueeze(1)
            t_tensor=x.new_full((config.n_samples,),fill_value=cur_t, dtype=torch.long)

            #eps = net(x, t_tensor,condition=condition)
            if scale==1:
                eps= net(x,x0, t_tensor, condition=condition)
            else:
                eps_theta1 = net(x,x0, t_tensor,condition=condition)
                eps_theta2 = net(x,x0, t_tensor,condition=None)
                eps = scale * eps_theta1 - (scale-1)*eps_theta2
                #eps = (1+scale) * eps_theta1 - scale * eps_theta2

            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur)**0.5 * x
            second_term = ((1 - ab_prev - var)**0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x = first_term + second_term + third_term

        return x

    def paint(self, x: torch.Tensor, cond: torch.Tensor,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None
              ):
        for t in range(self.n_steps-1, -1, -1):
            ts = x.new_full((1,),fill_value=t, dtype=torch.long)
            x_prev_known = self.q_sample(orig,ts)
            x_prev = self.p_sample(x,ts,condition=cond)
            if t!=0:
                x_prev = mask*x_prev_known + (1-mask)*x_prev
            x = x_prev
        return x
    
    def paint_ddim(self,
                      img_or_shape,
                      condition,
                      orig,
                      mask,
                      simple_var=True,
                      ddim_step=100,
                      eta=1):
        if simple_var:
            eta = 1
        device=config.device
        ts = torch.linspace(self.n_steps, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        net=self.eps_model
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)
        batch_size = x.shape[0]
        net = net.to(device)
        for i in tqdm(range(1, ddim_step + 1),
                      f'DDIM sampling with eta {eta} simple_var {simple_var}'):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            ab_cur = self.alpha_bar[cur_t]
            ab_prev = self.alpha_bar[prev_t] if prev_t >= 0 else 1

            #t_tensor = torch.tensor([cur_t] * batch_size, dtype=torch.long).to(device).unsqueeze(1)
            t_tensor=x.new_full((config.n_samples,),fill_value=cur_t, dtype=torch.long)
            eps = net(x, t_tensor,condition=condition)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur)**0.5 * x
            second_term = ((1 - ab_prev - var)**0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            #x = first_term + second_term + third_term
            x_fg = first_term + second_term + third_term
            x_bg = self.q_sample(orig,t_tensor)
            x = x_bg * mask + (1-mask)*x_fg

        return x

    
    def loss(self, x0: torch.Tensor, x1,noise: Optional[torch.Tensor] = None,condition:torch.tensor=None):
        """
        return:loss
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        
        eps_theta = self.eps_model(xt,x1, t,condition=condition)

        return F.mse_loss(noise, eps_theta)
    
    def classifire_p_sample(self, xt: torch.Tensor, t: torch.Tensor,condition:torch.tensor,scale=1.):
        if scale==1:
            eps_theta = self.eps_model(xt, t,condition=condition)
        else:
            eps_theta1 = self.eps_model(xt, t,condition=condition)
            eps_theta2 = self.eps_model(xt, t,condition=None)
            eps_theta = scale * eps_theta1 - (scale-1)*eps_theta2
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5

        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)

        var = gather(self.sigma2, t)

        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps


    def sample(self,n_samples,condition,image_channels,image_size,device):
        b,c,w,h = condition.shape

        with torch.no_grad():
            x = torch.randn([n_samples, image_channels, image_size, image_size],device=device)

            for t in range(self.n_steps-1, -1, -1):
                x = self.p_sample(x, x.new_full((n_samples,),fill_value=t, dtype=torch.long),condition=condition)
            return x

