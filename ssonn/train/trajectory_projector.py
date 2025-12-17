# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 02:21:24 2024
@author: Влад
"""
import math
import torch
from collections.abc import Iterable
import matplotlib.pyplot as plt

from ssonn.train.tools import color_by_id, mix_color


# Если рандомизацию надо зафиксировать, то делать это через torch.random.manual_seed, но можно, если приспичит, из dict тензоры вынуть.
class TrajectoryProjector:
    @staticmethod
    def radial(target:torch.Tensor, device, dtype):
        rnd: torch.Tensor = 2*math.pi*torch.rand_like(target).to(device)
        radials = [rnd.sin().to(dtype), rnd.cos().to(dtype)]
        return radials


    def __init__(self, model:torch.nn.Module or Iterable[torch.Tensor], device=torch.device('cpu'), dtype=torch.float16):
        if isinstance(model, torch.nn.Module):
            model = model.parameters()
        self.parameters = dict()
        self.trajectory = list()
        for p in model:
            self.parameters[p] = TrajectoryProjector.radial(p, device, dtype=torch.float16)
        self.shift = torch.zeros((2,), device=device, dtype=torch.float16) # Иногда бывает поллезно чтобы центровать проекцию

    def update(self, params: Iterable[torch.Tensor]):
        for p in params:
            self.parameters[p] = TrajectoryProjector.radial(p, torch.device('cpu'), dtype=torch.float16)

    # Вычисляем координаты проекции состояния сети
    def Projection(self:"TrajectoryProjector"):
        with torch.no_grad():
            # Тут ещё пакетный режим умножения foreach можно добавить, просто скопировав из оптимизаторов.
            # А то сейчас они тупо по одному умножаются, зато так память экономится. Потому что тензор конвертируется в другой
            # тип и если надо переносится на девайс только непосредственно перед умножением.
            # Если траекторию считать не каждый ход и держать множители на cpu это существенная прибавка к экономии памяти
            position = torch.stack([
                sum([key.view(-1).dot(value[0].to(key.dtype).to(key.device).view(-1)) for key, value in self.parameters.items()]),
                sum([key.view(-1).dot(value[1].to(key.dtype).to(key.device).view(-1)) for key, value in self.parameters.items()])
            ]).cpu()
            return self.shift.to(position.dtype).cpu() + position
        
    # Поместить очередную точку во внутреннее хранилище
    def CollectProjection(self:"TrajectoryProjector"):
        self.trajectory.append(self.Projection())
    
    def draw(self:"TrajectoryProjector", color:list=None, *, max_drawable_pices:int = None, part:slice = None, lr:float = None):
        '''
        color - Массив цветов от старой к новйо частям траектории
        part - Срез данных, можно например рисовать только конец траектории или каждую 10-ую точку
        max_drawable_pices - Максимальное количество кусков проприсовки, а то длинные тректории в десятки тысяч медленно рисуются.
        Будет выбран минимальный одинаковый шаг прореживания, чтобы в эти цифры уложиться.
        Может сильно загрублять информацию о дисперии, так что если используется такой способ для всей траектории,
        то небольшой последний участок траектории лучше отрисовать отдельно.
        '''
        if color is None:
            color = [color_by_id(0),color_by_id(1)]
        parts = len(color) - 1
        data = self.trajectory if part is None else self.trajectory[part]
        if (max_drawable_pices is not None) and (len(self.trajectory) > max_drawable_pices):
            data = data[::1+len(self.trajectory)//max_drawable_pices]
        length = len(data)
        for i,(bc,ec) in list(enumerate(zip(color[:-1],color[1:])))[::-1]:
            TrajectoryProjector.draw_trajectory(data[i*length//parts:(i+1)*length//parts+1], lr, ec, bc, rescale_field = (i == 0))
    

    @staticmethod
    def draw_trajectory(data:torch.Tensor or Iterable[torch.Tensor], lr:float = None, color=None, background='#ffffff', lr_color='#44FF44', parts:int=10, rescale_field:bool = True):
        if color is None:
            color = color_by_id(0)
        ax = plt.gca()
        if not isinstance(data, torch.Tensor):
            data = torch.stack(data)
        parts = min(parts, data.size(0))
        part = min(data.size(0)//parts,1)
        if (lr_color is not None) and (lr is not None):
            ax.add_patch(plt.Circle(data[-1], lr*3, fill = False, linewidth=1, edgecolor=lr_color))
            ax.add_patch(plt.Circle(data[-1], lr,  fill = False, linewidth=1, edgecolor=mix_color(lr_color, '#000000', .75)))
        plt.grid(True)
        for i in range(parts):
            block = data[i*data.size(0)//parts: (i+1)*data.size(0)//parts+1]
            plt.plot(block[:,0], block[:,1], color=mix_color(color, background, (i+1)/parts))
        bbox = ax.get_window_extent()
        scale = bbox.width/bbox.height
        if rescale_field:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            required_x = (ylim[1]-ylim[0]) * scale
            required_y = (xlim[1]-xlim[0]) / scale
            if xlim[1]-xlim[0] < required_x:
                delta_x = required_x - (xlim[1]-xlim[0])
                ax.set_xlim(xlim[0]-delta_x/2, xlim[1]+delta_x/2)
            elif ylim[1]-ylim[0] < required_y:
                delta_y = required_y - (ylim[1]-ylim[0])
                ax.set_ylim(ylim[0]-delta_y/2, ylim[1]+delta_y/2)
    
#model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).eval().cuda()
#projection = TrajectoryProjector(model, torch.device('cpu'))
#projection.Projection()
