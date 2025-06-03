# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021
@author: droes
"""
import keyboard 
import cv2 
import numpy as np 
import os 

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import ( 
    histogram_figure_numba, 
    calculate_stats, 
    calculate_entropy, 
    apply_equalization, 
    linear_transform,
    apply_gaussian_blur,
    apply_sharpen,
    apply_sobel_edges,
    apply_canny_edges,
    apply_gabor_filter,
    FACE_CASCADE, 
    load_filter_asset,
    overlay_transparent_asset,
    FILTER_ASSETS 
)

# --- Define some RGB colors ---
COLOR_GREEN = (0, 200, 0)    
COLOR_RED = (200, 0, 0)      
COLOR_BLUE = (0, 100, 200) # For general info like stats, contrast, brightness   

# --- Load Face Filter Assets ---
assets_dir = "filter_assets" 
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)
    print(f"Created directory: {assets_dir}. Please add your PNG filter assets there.")

asset_files_map = {
    "bunny1": "bunny1.png", "bunny2": "bunny2.png",
    "cat1": "cat1.png", "cat2": "cat2.png", 
    "devil1": "devil1.png",
    "dog1": "dog1.png", "dog2": "dog2.png", "dog3": "dog3.png",
    "sunglass1": "sunglass1.png", "sunglass2": "sunglass2.png",
    "sunglass3": "sunglass3.png", "sunglass4": "sunglass4.png"
}
for asset_name, file_name in asset_files_map.items():
    path = os.path.join(assets_dir, file_name)
    if os.path.exists(path):
        load_filter_asset(asset_name, path)
    else:
        print(f"!!! Asset file not found: {path}. Skipping. !!!")

FACE_FILTER_LIST = [
    "none", "bunny1", "bunny2", "cat1", "cat2", "devil1", 
    "dog1", "dog2", "dog3", "sunglass1", "sunglass2", "sunglass3", "sunglass4"
]


def custom_processing(img_source_generator):
    fig, ax, background, r_plot_line, g_plot_line, b_plot_line = initialize_hist_figure()
    
    apply_eq=False; lt_alpha=1.0; lt_beta=0.0; mirror_flip=True 
    is_blur=False; is_sharp=False; is_sobel=False; is_canny=False; is_gabor=False
    
    current_face_filter_idx = 0 
    debounce=0; DEBOUNCE_FRAMES=5 
    gabor_theta_idx=0; gabor_thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4] 

    def db(): nonlocal debounce; debounce = DEBOUNCE_FRAMES

    for seq_orig_rgb in img_source_generator: 
        proc_rgb = seq_orig_rgb.copy()

        if debounce > 0: debounce -= 1
        else: 
            if keyboard.is_pressed('e'): apply_eq=not apply_eq;print(f"Eq:{apply_eq}");db()
            if keyboard.is_pressed('f'): mirror_flip=not mirror_flip;print(f"Flip:{mirror_flip}");db()
            if keyboard.is_pressed('+')or keyboard.is_pressed('='):lt_beta+=5;db()
            if keyboard.is_pressed('-'):lt_beta-=5;db()
            if keyboard.is_pressed('page up'):lt_alpha+=0.1;db()
            if keyboard.is_pressed('page down'):lt_alpha-=0.1;db()
            if keyboard.is_pressed('0'):is_blur=is_sharp=is_sobel=is_canny=is_gabor=False;print("Img Filters OFF");db()
            if keyboard.is_pressed('1'):is_blur=not is_blur;print(f"Blur:{is_blur}");db()
            if keyboard.is_pressed('2'):is_sharp=not is_sharp;print(f"Sharp:{is_sharp}");db()
            if keyboard.is_pressed('3'):is_sobel=not is_sobel;print(f"Sobel:{is_sobel}");db()
            if keyboard.is_pressed('4'):is_canny=not is_canny;print(f"Canny:{is_canny}");db()
            if keyboard.is_pressed('5'):is_gabor=not is_gabor;print(f"Gabor:{is_gabor}");db()
            if is_gabor and keyboard.is_pressed('g'):gabor_theta_idx=(gabor_theta_idx+1)%len(gabor_thetas);db()
            if keyboard.is_pressed('p'):current_face_filter_idx=(current_face_filter_idx+1)%len(FACE_FILTER_LIST);print(f"FaceFilt:{FACE_FILTER_LIST[current_face_filter_idx]}");db()
        
        if mirror_flip: proc_rgb=cv2.flip(proc_rgb,1)
        proc_rgb=linear_transform(proc_rgb,alpha=lt_alpha,beta=lt_beta)
        if is_blur: proc_rgb=apply_gaussian_blur(proc_rgb)
        # Apply Equalization after blur, but before sharpen perhaps
        if apply_eq: proc_rgb=apply_equalization(proc_rgb)
        if is_sharp: proc_rgb=apply_sharpen(proc_rgb)
        
        temp_edge_input = proc_rgb.copy()
        if is_sobel: proc_rgb = apply_sobel_edges(temp_edge_input)
        elif is_canny: proc_rgb = apply_canny_edges(temp_edge_input)
        elif is_gabor: proc_rgb = apply_gabor_filter(temp_edge_input, theta=gabor_thetas[gabor_theta_idx])

        active_face_filter_key = FACE_FILTER_LIST[current_face_filter_idx]
        if active_face_filter_key != "none" and FACE_CASCADE and active_face_filter_key in FILTER_ASSETS:
            gray_for_face = cv2.cvtColor(seq_orig_rgb, cv2.COLOR_RGB2GRAY)
            if mirror_flip: gray_for_face = cv2.flip(gray_for_face, 1)
            faces = FACE_CASCADE.detectMultiScale(gray_for_face, 1.15, 5, minSize=(70,70))

            for (x,y,w,h) in faces:
                asset_to_apply = FILTER_ASSETS[active_face_filter_key]
                scale_factor = 1.0; x_offset = 0.5; y_offset = 0.5; y_anchor_is_top = False

                if "bunny" in active_face_filter_key or "cat" in active_face_filter_key or \
                   "dog" in active_face_filter_key or "devil" in active_face_filter_key:
                    if active_face_filter_key == "devil1":
                        asset_w = int(w*1.0); asset_h = int(asset_w * asset_to_apply.shape[0]/asset_to_apply.shape[1])
                        x_pos = x+(w-asset_w)//2; y_pos = y-int(asset_h*0.3) # Tuned for devil horns
                    else: # General ears/full face that are not sunglasses
                        asset_w = int(w * 1.1); asset_h = int(asset_w * asset_to_apply.shape[0]/asset_to_apply.shape[1])
                        x_pos = x+(w-asset_w)//2; y_pos = y+(h-asset_h)//2 - int(h*0.1) # Centered, slightly up
                elif "sunglasses" in active_face_filter_key:
                    asset_w = int(w*1.0); asset_h = int(asset_w * asset_to_apply.shape[0]/asset_to_apply.shape[1])
                    x_pos = x+(w-asset_w)//2; y_pos = y+int(h*0.30)-(asset_h//2) # Tuned for sunglasses
                else: # Fallback
                    asset_w = int(w*1.0); asset_h = int(asset_w*asset_to_apply.shape[0]/asset_to_apply.shape[1])
                    x_pos = x+(w-asset_w)//2; y_pos = y+(h-asset_h)//2
                
                if asset_w > 0 and asset_h > 0 :
                    asset_resized = cv2.resize(asset_to_apply, (asset_w, asset_h))
                    proc_rgb = overlay_transparent_asset(proc_rgb, asset_resized, x_pos, y_pos)
        
        stats=calculate_stats(seq_orig_rgb); entropy=calculate_entropy(seq_orig_rgb)
        texts=[]; colors=[]; info_color=COLOR_BLUE # Default info color is Blue
        
        # Stats line (always blue)
        texts.append(f"M:{stats['mean']:.1f} SD:{stats['std_dev']:.1f} Ma:{stats['max']} Mi:{stats['min']} Mo:{stats['mode']} E:{entropy:.1f}")
        colors.append(info_color)
        
        # Equalization status (dynamic color + key hint)
        texts.append(f"Equalization (e): {'ON' if apply_eq else 'OFF'}")
        colors.append(COLOR_GREEN if apply_eq else COLOR_RED)

        # Mirror Flip status (dynamic color + key hint)
        texts.append(f"Mirror Flip (f): {'ON' if mirror_flip else 'OFF'}")
        colors.append(COLOR_GREEN if mirror_flip else COLOR_RED)

        # Contrast and Brightness (always blue + key hints)
        texts.append(f"Contrast (PgUp/Dn): {lt_alpha:.1f}")
        colors.append(info_color)
        texts.append(f"Brightness (+/-): {lt_beta}")
        colors.append(info_color)
        
        # Image Filters status line
        active_img_filters=[]
        if is_blur:active_img_filters.append("Blu(1)")
        if is_sharp:active_img_filters.append("Shr(2)")
        if is_sobel:active_img_filters.append("Sob(3)")
        if is_canny:active_img_filters.append("Can(4)")
        if is_gabor:active_img_filters.append(f"Gab(5)(th:{gabor_thetas[gabor_theta_idx]:.0f})")
        img_filter_text_line = "ImgFilt(0-Off): " 
        if active_img_filters:
            img_filter_text_line += ",".join(active_img_filters)
        else:
            img_filter_text_line += "None"
        texts.append(img_filter_text_line)
        colors.append(COLOR_GREEN if active_img_filters else info_color)
        
        # Face Filter status line
        face_filter_text_line = f"FaceFilt(p): {FACE_FILTER_LIST[current_face_filter_idx].capitalize()}"
        texts.append(face_filter_text_line)
        colors.append(COLOR_GREEN if FACE_FILTER_LIST[current_face_filter_idx]!="none" else info_color)

        r_bar_data, g_bar_data, b_bar_data = histogram_figure_numba(proc_rgb)
        update_histogram(fig, ax, background, r_plot_line, g_plot_line, b_plot_line, 
                         r_bar_data, g_bar_data, b_bar_data)
        
        final_img=proc_rgb.copy() 
        final_img=plot_overlay_to_image(final_img,fig)
        final_img=plot_strings_to_image(final_img,texts,list_of_colors=colors, right_space=600, top_space=15, line_height=20, font_scale=0.5)
        yield final_img

def main():
    w,h,f = 1280,720,30; vc=VirtualCamera(f,w,h)
    print("--- Controls ---")
    print(" e:Eq | f:Flip | +/-:Bright | PgUp/Dn:Contrast | p:Cycle FaceFilt")
    print(" ImgFilt: 0-All OFF | 1-Blur | 2-Sharp | 3-Sobel | 4-Canny | 5-Gabor")
    print(" g: Cycle Gabor Theta (if Gabor active)")
    print(" q:Quit"); print("----------------")
    vc.virtual_cam_interaction(custom_processing(vc.capture_cv_video(0,bgr_to_rgb=True)))

if __name__ == "__main__":
    main()