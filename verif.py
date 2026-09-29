import numpy as np
import sympy
import cv2
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_spatial_equation(eq_str, feature_dict, shape):
    if not eq_str or eq_str == "None":
        return np.zeros(shape, dtype=np.uint8)

    try:
        expr = sympy.sympify(eq_str, locals={'max': sympy.Max, 'min': sympy.Min})
        
        # Dynamically map symbols based on available features
        symbols = [sympy.Symbol(k) for k in feature_dict.keys()]
        
        func = sympy.lambdify(symbols, expr, modules=[
            {'square': np.square, 'abs': np.abs, 'sqrt': np.sqrt, 
             'sin': np.sin, 'cos': np.cos, 'max': np.maximum, 'min': np.minimum, 
             'tanh': np.tanh, 'exp': np.exp}, 
            'numpy'
        ])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pred = func(**feature_dict)
            if np.isscalar(pred): 
                pred = np.full(list(feature_dict.values())[0].shape, pred)
            
            # Clean non-finite numbers before multiplier scaling
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            pred = pred * 255.0
            
        pred = np.clip(pred, 0, 255)
        return pred.reshape(shape).astype(np.uint8)
        
    except Exception as e:
        print(f"Error evaluating equation: {e}")
        return np.zeros(shape, dtype=np.uint8)

def calculate_metrics(orig_channel, recon_channel):
    """Calculates extended statistics."""
    if orig_channel is None or recon_channel is None:
        return {'mae': 0, 'peak': 0, 'median': 0, 'psnr': 0}
    
    orig_f = orig_channel.astype(np.float64)
    recon_f = recon_channel.astype(np.float64)
    
    diff = np.abs(orig_f - recon_f)
    
    stats = {
        'mae': np.mean(diff),
        'peak': np.max(diff),
        'median': np.median(diff),
        'psnr': 0
    }
    
    mse = np.mean(diff ** 2)
    stats['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
    
    return stats

def verify_and_reconstruct(original_path, equations, output_stem="reconstructed_fume", fourier_freqs=0):
    img = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    if img is None: return {}

    h, w = img.shape[:2]
    
    channels = {}
    if img.ndim == 2:
        channels['b'] = channels['g'] = channels['r'] = img
    elif img.shape[2] == 4:
        channels['b'], channels['g'], channels['r'], channels['a'] = cv2.split(img)
    else:
        channels['b'], channels['g'], channels['r'] = cv2.split(img)

    EPSILON = 1e-5
    x_coords = np.linspace(0 + EPSILON, 1 + EPSILON, w)
    y_coords = np.linspace(0 + EPSILON, 1 + EPSILON, h)
    xv, yv = np.meshgrid(x_coords, y_coords)
    x_flat, y_flat = xv.flatten(), yv.flatten()

    # Reconstruct the identical sequential feature dictionary
    feature_dict = {}
    idx = 0
    feature_dict[f'x{idx}'] = x_flat; idx += 1
    feature_dict[f'x{idx}'] = y_flat; idx += 1
    
    if fourier_freqs > 0:
        for k in range(fourier_freqs):
            freq = (2 ** k) * 2 * np.pi
            feature_dict[f'x{idx}'] = np.sin(x_flat * freq); idx += 1
            feature_dict[f'x{idx}'] = np.cos(x_flat * freq); idx += 1
            feature_dict[f'x{idx}'] = np.sin(y_flat * freq); idx += 1
            feature_dict[f'x{idx}'] = np.cos(y_flat * freq); idx += 1

    recon_channels = {}
    stats_results = {}
    
    for key, img_key in [('B','b'), ('G','g'), ('R','r'), ('A','a')]:
        if img_key in channels:
            eq_data = equations.get(key)
            if isinstance(eq_data, dict):
                eq_str = eq_data.get('equation')
            else:
                eq_str = eq_data

            recon_channels[img_key] = evaluate_spatial_equation(eq_str, feature_dict, (h, w))
            stats_results[key] = calculate_metrics(channels[img_key], recon_channels[img_key])
        else:
            recon_channels[img_key] = None
            if img_key == 'a': recon_channels['a'] = np.full((h, w), 255, dtype=np.uint8)

    merge_list = [recon_channels['b'], recon_channels['g'], recon_channels['r']]
    if 'a' in channels: merge_list.append(recon_channels['a'])
    
    cv2.imwrite(f"{output_stem}.png", cv2.merge(merge_list))
    
    diff_src = img if 'a' in channels else img[:,:,:3]
    diff_rec = cv2.merge(merge_list) if 'a' in channels else cv2.merge(merge_list[:3])
    diff = cv2.absdiff(diff_src, diff_rec)
    cv2.imwrite(f"{output_stem}_error.png", np.clip(np.max(diff, axis=2) * 4, 0, 255).astype(np.uint8))
    
    return stats_results