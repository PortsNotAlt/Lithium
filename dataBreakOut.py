#
# Breaks out colour data and generates Normalized Spatial Coordinates (0.0 to 1.0)
# Author: Port Christman
#
import numpy as np
import pandas as pd
import cv2
import os

def process_media(input_path, output_stem_path=None, chunk_size=None, 
                  max_workers=None, return_data=False, verbose=False, 
                  max_dimension=64, fourier_freqs=0):
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        cap = cv2.VideoCapture(input_path)
        ret, img = cap.read()
        cap.release()
        if not ret: raise ValueError("Could not read media.")

    h, w = img.shape[:2]
    scale = min(max_dimension / h, max_dimension / w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        if verbose: print(f"Resizing input from {w}x{h} to {new_w}x{new_h}...")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    h, w = img.shape[:2]
    
    if img.ndim == 2:
        b, g, r = img, img, img
        a = None
    elif img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
    else:
        b, g, r = cv2.split(img)
        a = None

    EPSILON = 1e-5
    x_coords = np.linspace(0 + EPSILON, 1 + EPSILON, w)
    y_coords = np.linspace(0 + EPSILON, 1 + EPSILON, h)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    x_flat = xv.flatten()
    y_flat = yv.flatten()
    
    data = {}
    
    def package_df(channel_data):
        df_dict = {}
        idx = 0
        df_dict[f'x{idx}'] = x_flat; idx += 1
        df_dict[f'x{idx}'] = y_flat; idx += 1
        
        # Inject pre-computed Fourier octave features sequentially
        if fourier_freqs > 0:
            for k in range(fourier_freqs):
                freq = (2 ** k) * 2 * np.pi
                df_dict[f'x{idx}'] = np.sin(x_flat * freq); idx += 1
                df_dict[f'x{idx}'] = np.cos(x_flat * freq); idx += 1
                df_dict[f'x{idx}'] = np.sin(y_flat * freq); idx += 1
                df_dict[f'x{idx}'] = np.cos(y_flat * freq); idx += 1
                
        df_dict['y'] = channel_data.flatten() / 255.0
        return pd.DataFrame(df_dict)

    # ADD THESE MISSING LINES:
    data['r'] = package_df(r)
    data['g'] = package_df(g)
    data['b'] = package_df(b)
    
    if a is not None: 
        data['a'] = package_df(a)

    return data