from numba import njit
import numpy as np
import cv2 

# --- Face Detection Global (Load once) ---
# Download 'haarcascade_frontalface_default.xml' from OpenCV's GitHub:
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# Place it in your project directory (e.g., D:\MyProjects\cv_project\VirtualCamera\)
try:
    # Try loading from standard OpenCV data path first
    face_cascade_path_cv = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    # Try loading from local project directory as a fallback
    face_cascade_path_local = 'haarcascade_frontalface_default.xml'

    if cv2.os.path.exists(face_cascade_path_cv):
        FACE_CASCADE = cv2.CascadeClassifier(face_cascade_path_cv)
        print("Loaded Haar Cascade from OpenCV data path.")
    elif cv2.os.path.exists(face_cascade_path_local):
        FACE_CASCADE = cv2.CascadeClassifier(face_cascade_path_local)
        print("Loaded Haar Cascade from local project directory.")
    else:
        FACE_CASCADE = None # Initialize to None if not found immediately

    if FACE_CASCADE is None or FACE_CASCADE.empty(): # Check if loading failed
        print("!!! WARNING: HAAR CASCADE FOR FACE DETECTION NOT LOADED !!!")
        print(f"!!! Attempted paths: '{face_cascade_path_cv}' and '{face_cascade_path_local}' !!!")
        print("!!! Make sure 'haarcascade_frontalface_default.xml' is accessible. !!!")
        FACE_CASCADE = None # Explicitly set to None if empty
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    FACE_CASCADE = None

# --- Filter Asset Loading Global (Load once) ---
FILTER_ASSETS = {} # Dictionary to store loaded assets (name: image_data)

def load_filter_asset(asset_name, file_path):
    """Loads a filter asset (PNG with transparency) and stores it."""
    global FILTER_ASSETS
    try:
        # IMREAD_UNCHANGED ensures the alpha channel is loaded if present
        asset = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) 
        if asset is None:
            print(f"!!! WARNING: Could not load filter asset: {file_path} !!!")
            return
        
        # Ensure the asset has 4 channels (B, G, R, Alpha)
        if asset.shape[2] != 4:
            print(f"!!! WARNING: Filter asset {file_path} does not have 4 channels (no alpha). Creating opaque alpha. !!!")
            # If not 4 channels, assume it's BGR and add a fully opaque alpha channel
            if asset.shape[2] == 3:
                b, g, r = cv2.split(asset)
                alpha_channel = np.ones(b.shape, dtype=b.dtype) * 255 # Opaque alpha
                asset = cv2.merge((b, g, r, alpha_channel))
            else:
                print(f"!!! Asset {file_path} has an unexpected number of channels: {asset.shape[2]}. Skipping. !!!")
                return

        FILTER_ASSETS[asset_name] = asset
        print(f"Loaded filter asset: {asset_name} with shape {asset.shape}")
    except Exception as e:
        print(f"Error loading filter asset {file_path}: {e}")

def overlay_transparent_asset(background_rgb, overlay_rgba_bgr, x_offset, y_offset):
    """
    Overlays a transparent RGBA asset (loaded by OpenCV as BGRA) onto an RGB background.
    background_rgb: The RGB background image.
    overlay_rgba_bgr: The BGRA overlay image (from cv2.imread with IMREAD_UNCHANGED).
    x_offset, y_offset: Top-left coordinates where the overlay should be placed on the background.
    """
    bg_h, bg_w, _ = background_rgb.shape
    overlay_h, overlay_w, _ = overlay_rgba_bgr.shape

    # Get the alpha channel from the overlay and normalize it to 0-1
    # OpenCV loads as BGRA, so alpha is the 4th channel (index 3)
    alpha_mask = overlay_rgba_bgr[:, :, 3] / 255.0 
    # Get the color channels (BGR) from the overlay
    overlay_colors_bgr = overlay_rgba_bgr[:, :, :3]

    # Ensure offsets are integers for slicing
    x_offset, y_offset = int(x_offset), int(y_offset)

    # Define the region of interest (ROI) on the background
    y1, y2 = max(0, y_offset), min(bg_h, y_offset + overlay_h)
    x1, x2 = max(0, x_offset), min(bg_w, x_offset + overlay_w)

    # Define the region of the overlay to use
    overlay_y1, overlay_y2 = max(0, -y_offset), overlay_h - max(0, (y_offset + overlay_h) - bg_h)
    overlay_x1, overlay_x2 = max(0, -x_offset), overlay_w - max(0, (x_offset + overlay_w) - bg_w)

    if y1 >= y2 or x1 >= x2 or overlay_y1 >= overlay_y2 or overlay_x1 >= overlay_x2:
        return background_rgb # ROI or overlay part is invalid

    # Extract the relevant part of the overlay and its alpha mask
    alpha_sub_mask = alpha_mask[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    overlay_colors_bgr_sub = overlay_colors_bgr[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    # Convert overlay BGR colors to RGB to match the background_rgb
    overlay_colors_rgb_sub = cv2.cvtColor(overlay_colors_bgr_sub, cv2.COLOR_BGR2RGB)
    
    # Extract the ROI from the background
    roi = background_rgb[y1:y2, x1:x2]

    # Blend using the alpha mask (channel by channel)
    for c in range(3): # For R, G, B channels
        roi[:, :, c] = (alpha_sub_mask * overlay_colors_rgb_sub[:, :, c] +
                        (1 - alpha_sub_mask) * roi[:, :, c])
    
    # Place the blended ROI back into the background image
    background_rgb[y1:y2, x1:x2] = roi
    return background_rgb


# --- Existing functions from basics.py ---
@njit
def histogram_figure_numba(np_img_rgb): 
    h, w, c = np_img_rgb.shape
    r_hist = np.zeros(256, dtype=np.int32)
    g_hist = np.zeros(256, dtype=np.int32)
    b_hist = np.zeros(256, dtype=np.int32)
    for y_idx in range(h): 
        for x_idx in range(w):
            r_hist[np_img_rgb[y_idx, x_idx, 0]] += 1 
            g_hist[np_img_rgb[y_idx, x_idx, 1]] += 1 
            b_hist[np_img_rgb[y_idx, x_idx, 2]] += 1 
    max_val_r = np.max(r_hist) if np.max(r_hist) > 0 else 1.0
    max_val_g = np.max(g_hist) if np.max(g_hist) > 0 else 1.0
    max_val_b = np.max(b_hist) if np.max(b_hist) > 0 else 1.0
    norm_factor = 3.0 
    r_bars = (r_hist / max_val_r) * norm_factor
    g_bars = (g_hist / max_val_g) * norm_factor
    b_bars = (b_hist / max_val_b) * norm_factor
    return r_bars, g_bars, b_bars

def calculate_stats(np_img_rgb):
    gray_img = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2GRAY) 
    mean_val = np.mean(gray_img); std_dev_val = np.std(gray_img)
    max_val = np.max(gray_img); min_val = np.min(gray_img)
    counts = np.bincount(gray_img.ravel()); mode_val = np.argmax(counts) if len(counts) > 0 else 0 
    return {"mean": mean_val, "std_dev": std_dev_val, "max": max_val, "min": min_val, "mode": mode_val}
    
def linear_transform(np_img_rgb, alpha=1.0, beta=0.0): 
    return cv2.convertScaleAbs(np_img_rgb, alpha=alpha, beta=beta)

def calculate_entropy(np_img_rgb):
    gray_img = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    if hist.sum() == 0: return 0.0
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    if len(hist_norm) == 0: return 0.0
    return -np.sum(hist_norm * np.log2(hist_norm))

def apply_equalization(np_img_rgb):
    ycrcb_img = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)

def apply_gaussian_blur(np_img_rgb, kernel_size_val=15):
    k_val = kernel_size_val if kernel_size_val % 2 == 1 else kernel_size_val + 1
    return cv2.GaussianBlur(np_img_rgb, (k_val, k_val), 0)

def apply_sharpen(np_img_rgb):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(np_img_rgb, -1, kernel)

def apply_sobel_edges(np_img_rgb):
    gray_img = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_norm = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(sobel_norm, cv2.COLOR_GRAY2RGB)

def apply_canny_edges(np_img_rgb, threshold1=100, threshold2=200):
    gray_img = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_img, threshold1, threshold2) 
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_gabor_filter(np_img_rgb, ksize=31, sigma=4.0, theta=np.pi/4, lambd=10.0, gamma=0.5, psi=0):
    gray_img = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2GRAY)
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_f = cv2.filter2D(gray_img.astype(np.float32)/255.0, cv2.CV_32F, gabor_kernel) 
    filtered_norm = cv2.normalize(filtered_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(filtered_norm, cv2.COLOR_GRAY2RGB)