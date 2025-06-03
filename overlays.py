# overlays.py

import numpy as np
import cv2 
from matplotlib import pyplot as plt 


def initialize_hist_figure():
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlim([-0.5, 255.5])
    ax.set_ylim([0,3])
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    def_x_line = np.arange(0, 256, 1)
    r_plot = ax.plot(def_x_line, def_x_line, 'r', animated=True)[0]
    g_plot = ax.plot(def_x_line, def_x_line, 'g', animated=True)[0]
    b_plot = ax.plot(def_x_line, def_x_line, 'b', animated=True)[0]
    return fig, ax, background, r_plot, g_plot, b_plot


def update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars):
    fig.canvas.restore_region(background)        
    r_plot.set_ydata(r_bars)        
    g_plot.set_ydata(g_bars)        
    b_plot.set_ydata(b_bars)
    ax.draw_artist(r_plot)
    ax.draw_artist(g_plot)
    ax.draw_artist(b_plot)
    fig.canvas.blit(ax.bbox)
    
    
def plot_overlay_to_image(np_img, plt_figure):
    rgba_buf = plt_figure.canvas.buffer_rgba()
    (w, h) = plt_figure.canvas.get_width_height()
    imga = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(h,w,4)[:,:,:3] 
    
    overlay_h, overlay_w = imga.shape[:2]
    target_h, target_w = np_img.shape[:2]
    
    h_end = min(overlay_h, target_h)
    w_end = min(overlay_w, target_w)

    imga_region = imga[:h_end, :w_end]
    mask_region = np.all(imga_region == [255, 255, 255], axis=-1) 
    
    np_img[:h_end, :w_end][~mask_region] = imga_region[~mask_region]
    return np_img


# +++ MODIFIED FUNCTION SIGNATURE +++
def plot_strings_to_image(np_img, list_of_strings, list_of_colors=None, 
                          default_text_color_rgb=(0, 200, 0), 
                          right_space=400, top_space=50,
                          line_height=25, font_scale=0.7, thickness=2): # Added new parameters
    '''
    Plots the string parameters below each other, starting from top right.
    np_img is assumed to be in RGB format.
    '''
    y_start = top_space
    (h, w, c) = np_img.shape

    if w < right_space : # Simplified width check
        # print('Warning: Image too small in width to print additional text.')
        return np_img 
        
    y_pos = y_start
    x_pos = max(10, w - right_space) 

    for i, text in enumerate(list_of_strings):
        if y_pos >= h - line_height : # Check if text will go out of bounds vertically
            break
        
        current_color_rgb = default_text_color_rgb
        if list_of_colors and i < len(list_of_colors):
            current_color_rgb = list_of_colors[i]
        
        cv2.putText(np_img, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, current_color_rgb, thickness, cv2.LINE_AA)
        y_pos += line_height
    
    return np_img