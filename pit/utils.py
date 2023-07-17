"""
Symbolic meaning of the shapes
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, Wedge, Arrow

COLOR_MAPS = [ 'white', 'black', 'gray', 'red', 'orange', 'yellow', 'green', 'blue', 'navy', 'purple', ]

def visualize_shapes(seq, ratio=1.0):
    N = len(seq)
    fig, ax = plt.subplots(1,1, figsize=(N*ratio,1.2*ratio))
    for i in range(N):
        shape = seq[i][0]
        color = seq[i][1]
        globals()[f'Shape{shape}'](x_offset=i*0.8, ax=ax, facecolor=COLOR_MAPS[color], edgecolor='black')
    ax.set_xlim(-0.5,N-1)
    ax.set_ylim(-0.5,0.5)
    ax.axis("off")
    return ax 

# -------------------------------------------------------
# Determine how to visualize shapes 
# -------------------------------------------------------

class Shape0:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0.2
        pts = np.array([[0.3,0], [-0.3,0], [0,-0.4]])
        pts[:,0] += x_offset
        pts[:,1] += y_offset
        p = Polygon(pts, **kwargs)
        ax.add_patch(p)

class Shape1:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0.2
        pts = np.array([[0.3,-0.4], [-0.3,-0.4], [0,0]])
        pts[:,0] += x_offset
        pts[:,1] += y_offset
        p = Polygon(pts,  **kwargs)
        ax.add_patch(p)

class Shape2:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0.05
        p = Rectangle([x_offset-0.25,y_offset-0.25], 0.4, 0.4, **kwargs)
        ax.add_patch(p)


class Shape3:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0 
        p = Circle([x_offset,y_offset], 0.25, **kwargs)
        ax.add_patch(p)

class Shape4:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0 
        p = Wedge([x_offset,y_offset], 0.25, 0, 300, **kwargs)
        ax.add_patch(p)

class Shape5:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0 
        p = Wedge([x_offset,y_offset], 0.25, 60, 0, **kwargs)
        ax.add_patch(p)

class Shape6:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0 
        p = Wedge([x_offset,y_offset], 0.25, 120, 60, **kwargs)
        ax.add_patch(p)

class Shape7:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0 
        p = Wedge([x_offset,y_offset], 0.25, 180, 120, **kwargs)
        ax.add_patch(p)

class Shape8:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0 
        p = Wedge([x_offset,y_offset], 0.25, 240, 180, **kwargs)
        ax.add_patch(p)

class Shape9:
    def __init__(self, x_offset, ax, **kwargs) -> None:
        y_offset = 0 
        p = Wedge([x_offset,y_offset], 0.25, 300, 240, **kwargs)
        ax.add_patch(p)





