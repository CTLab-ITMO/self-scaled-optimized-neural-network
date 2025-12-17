# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 23:51:42 2024
@author: Влад
"""
import matplotlib as mpl
import torch
import time
import math

# Инструменты по работе с временем
def time_to_str(value:float):
    ''' Преобразование float в краткую запись количества времени
        Args:
            value (float): Время в секундах
        Returns:
            str: Два самых главных компонента времени
    '''
    if value < 0.5:
        return f"{value*1000:.3f}:ms";
    t = []
    value = round(value)
    if value > 0: t.append(value%60); value = value//60
    if value > 0: t.append(value%60); value = value//60
    if value > 0: t.append(value%24); value = value//24
    if value > 0: t.append(value);
    t = [f'{v:02}{e}' for v,e in zip(t,['s','m','h','d'])]
    return ':'.join(t[-2:][::-1])

class stopwatch:
    ''' Просто класс, упрощающий контроль за временем, чтобы всякие сложные преобразования строк спрятать под капот
        Методы:
            elapsed():float
            progress(current:float, total:float):str
    '''
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self):
        return time.time() - self.start_time
    def progress(self, current, total):
        elapsed = self.elapsed()
        return f"{time_to_str(elapsed)}/{time_to_str(elapsed*total/current)}"
    def __repr__(self):
        return time_to_str(self.elapsed())

# Инструменты по работе с цветами
def color_by_id(color:int):
    ''' Вынимает цвета по айдишнику из стандартной матплотлибовской таблицы
        Args:
            color (int): Первый цвет в формате #RRGGBB
        Returns:
            str: цвет в формате #RRGGBB
    '''
    if isinstance(color, int):
        colors = mpl.rcParams["axes.prop_cycle"]
        return [c['color'] for c in colors][color % len(colors)]
    return color

def mix_color(color1:str, color2:str, proportion:float):
    ''' Смешивает два цвета в пропорции 
        Args:
            color1 (string): Первый цвет в формате #RRGGBB
            color2 (string): Второй цвет в формате #RRGGBB
            proportion (float): 1-целиком первый, 0-целикоми второй
        Returns:
            str: цвет в формате #RRGGBB
    '''
    c1_rgb = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
    c2_rgb = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))
    blended_rgb = (
        int(proportion * c1_rgb[0] + (1 - proportion) * c2_rgb[0]),
        int(proportion * c1_rgb[1] + (1 - proportion) * c2_rgb[1]),
        int(proportion * c1_rgb[2] + (1 - proportion) * c2_rgb[2]),
    )
    return '#{:02X}{:02X}{:02X}'.format(*blended_rgb)

def mix_color_white(color:str, proportion:float = 0.5):
    ''' Примешивает белый цвет
        Args:
            color (int): Первый цвет в формате #RRGGBB
            proportion (float)=0.5: 1 - чистый белый
        Returns:
            str: цвет в формате #RRGGBB
    '''
    return mix_color('#FFFFFF', color, proportion);

# Инструменты для манипуляций с сетью
def parameters_count(model:torch.nn.Module):
    '''Количество параметров в модели'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def zeros(model:torch.nn.Module):
    '''Количество нулей в модели'''
    return sum([(p==0).sum() for p in model.parameters()]).cpu()
def none_zeros(model:torch.nn.Module):
    '''Количество ненулевых значений в модели'''
    return parameters_count(model) - zeros(model)

def norma(a:list[torch.Tensor]):
    '''Норма листа векторов, рассматриваемого как один большой вектор'''
    return sum([(t.square().sum() if t is not None else 0) for t in a]).sqrt()
def angle_between(a:list[torch.Tensor], b:list[torch.Tensor], a_norma:float = None, b_norma:float = None, tensor:bool=True):
    '''Угол между векторами каждый из которых это лист тензоров'''
    a_norma = norma(a) if a_norma is None else a_norma
    b_norma = norma(b) if b_norma is None else b_norma
    dot_product = sum([(t1*t2).sum() if (t1 is not None) and (t2 is not None) else 0 for t1,t2 in zip(a,b)])
    angle = torch.clamp(dot_product/a_norma/b_norma,-1,1).acos()*(180/math.pi)
    if not tensor and isinstance(angle, torch.Tensor):
        angle = angle.cpu().item()
    return angle

def inspect(target):
    return set(dir(target))-set(dir(object))

class LogModule(torch.nn.Module):
    '''Модуль логирующий проходящие сквозь него данные.'''
    def __init__(self, label='', module:torch.nn.Module = None):
        super(LogModule, self).__init__()
        self.label=label
        self.module = module
    def forward(self, x:torch.Tensor or list[torch.Tensor]):
        y = x if self.module is None else self.module(x)
        if isinstance(x,torch.Tensor):
            print(f'{self.label}: {x.shape}, mean/std:{x.mean().item():.4f}/{x.std().item():.4e}{("" if self.module is None else f" => {y.shape}")}')
        else: # Iterable
            print(f'{self.label}: {", ".join([str([s for s in t.shape]) for t in x])}{("" if self.module is None else f" => {y.shape}")}')
        return y


def Test():
    print('time_to_str({1.6}):', time_to_str(1.6))
    print('time_to_str({1116}):', time_to_str(111.6))
    print('time_to_str({111116}):', time_to_str(111111.6))
    timer = stopwatch()
    timer.start_time -= 8
    print('stopwatch().timer.progress(1, 2):', timer.progress(1, 2))
    
    print('color_by_id(1):', color_by_id(1))
    print('mix_color(color_by_id(1), color_by_id(2), .5):', mix_color(color_by_id(1), color_by_id(2), .5))
    print('mix_color_white(color_by_id(1), .85):', mix_color_white(color_by_id(1), .85))