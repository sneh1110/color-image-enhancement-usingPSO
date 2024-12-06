import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.measure import shannon_entropy
import pywt
import pandas as pd

# Helper function: Resize images to match dimensions
def resize_to_original(enhanced, original):
    return cv2.resize(enhanced, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)

# Helper function: Display image comparisons
def display_comparisons(original, enhanced_list, method_names):
    plt.figure(figsize=(15, 10))
    total = len(enhanced_list)
    for i, (enhanced, method) in enumerate(zip(enhanced_list, method_names)):
        plt.subplot(2, total, i + 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')
        plt.subplot(2, total, total + i + 1)
        plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        plt.title(method)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Helper function: Plot histogram comparison
def plot_histogram_comparison(original, enhanced_list, method_names):
    plt.figure(figsize=(15, 10))
    for i, (enhanced, method) in enumerate(zip(enhanced_list, method_names)):
        plt.subplot(2, len(enhanced_list), i + 1)
        color = ('b', 'g', 'r')
        for j, col in enumerate(color):
            hist_orig = cv2.calcHist([original], [j], None, [256], [0, 256])
            plt.plot(hist_orig, color=col)
        plt.title("Original")
        plt.subplot(2, len(enhanced_list), len(enhanced_list) + i + 1)
        for j, col in enumerate(color):
            hist_enh = cv2.calcHist([enhanced], [j], None, [256], [0, 256])
            plt.plot(hist_enh, color=col)
        plt.title(method)
    plt.tight_layout()
    plt.show()

# 1. Fourier Transform-Based Enhancement
# 1. Fourier Transform-Based Enhancement
def fourier_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = fftshift(fft2(gray))
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create high-pass filter
    mask = np.ones((rows, cols), dtype=np.float32)
    mask[crow - 20:crow + 20, ccol - 20:ccol + 20] = 0  # Block low frequencies
    
    # Apply the mask and perform the inverse FFT
    f_transform_filtered = f_transform * mask
    inverse_transform = ifft2(ifftshift(f_transform_filtered))
    enhanced_image = np.abs(inverse_transform)
    
    # Normalize and scale the image
    enhanced_image = (enhanced_image / np.max(enhanced_image) * 255).astype(np.uint8)
    
    # Blend the enhanced image with the original for better visual results
    blended_image = cv2.addWeighted(gray, 0.5, enhanced_image, 0.5, 0)
    return cv2.cvtColor(blended_image, cv2.COLOR_GRAY2BGR)

# 2. Wavelet Transform-Based Enhancement

def wavelet_enhancement(image):
    # Perform 2D Discrete Wavelet Transform (DWT) using Haar wavelet
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs  # Approximation and detail coefficients
    
    # Enhance the detail coefficients by scaling them
    cH *= 1.5
    cV *= 1.5
    cD *= 1.5
    
    # Perform the inverse 2D DWT to get the enhanced image
    enhanced_image = pywt.idwt2((cA, (cH, cV, cD)), 'haar').astype(np.float32)
    
    # Normalize the output to the range [0, 255] and convert to uint8
    enhanced_image = np.uint8(np.clip(enhanced_image, 0, 255))
    
    return enhanced_image
 


# 3. Color Transformations (HSV, YCbCr, Lab)
def color_transform_enhancement(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    enhanced_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycbcr[:, :, 0] = cv2.equalizeHist(ycbcr[:, :, 0])
    enhanced_ycbcr = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    enhanced_lab = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    return enhanced_hsv, enhanced_ycbcr, enhanced_lab

# 4. Particle Swarm Optimization (PSO)-Based Enhancement
def pso_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Objective function: Combined metric of Laplacian variance, entropy, and SSIM
    def objective_function(params):
        alphas, betas = params[:, 0], params[:, 1]
        scores = []
        for alpha, beta in zip(alphas, betas):
            if alpha <= 0 or beta < -50:  # Slightly adjust beta to allow small negative brightness adjustments
                scores.append(float('inf'))
                continue
            enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

            # Apply pixel value clipping after enhancement
            enhanced = np.clip(enhanced, 0, 255)  # Clip to prevent over-exposure or underexposure

            laplacian_var = cv2.Laplacian(enhanced, cv2.CV_64F).var()
            entropy = shannon_entropy(enhanced)
            ssim_val = ssim(gray, enhanced)
            score = - (0.5 * laplacian_var + 0.3 * entropy + 0.2 * ssim_val)  # Weighted metric
            scores.append(score)
        return np.array(scores)

    # Parameters
    num_particles = 20
    num_iterations = 100
    early_stop_threshold = 1e-6  # Early stopping threshold for global best improvement
    alpha_bounds = (0.5, 2.0)  # Adjust contrast range to moderate values (e.g., 0.5 to 2.0)
    beta_bounds = (-50, 50)     # Brightness adjustment range: small positive or negative

    # Initialize particles and velocities
    particles = np.random.uniform([alpha_bounds[0], beta_bounds[0]],
                                  [alpha_bounds[1], beta_bounds[1]], (num_particles, 2))
    velocities = np.zeros_like(particles)
    personal_best = particles.copy()
    personal_best_scores = objective_function(personal_best)
    global_best = personal_best[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # PSO hyperparameters
    w_max, w_min = 0.9, 0.4  # Inertia weight range
    c1_max, c1_min = 2.5, 1.5  # Cognitive parameter range
    c2_max, c2_min = 1.5, 2.5  # Social parameter range

    for t in range(num_iterations):
        # Linearly adjust hyperparameters
        w = w_max - (w_max - w_min) * (t / num_iterations)
        c1 = c1_max - (c1_max - c1_min) * (t / num_iterations)
        c2 = c2_min + (c2_max - c2_min) * (t / num_iterations)

        # Update velocities and positions
        r1, r2 = np.random.rand(num_particles, 1), np.random.rand(num_particles, 1)
        velocities = (
            w * velocities
            + c1 * r1 * (personal_best - particles)
            + c2 * r2 * (global_best - particles)
        )
        particles += velocities
        particles[:, 0] = np.clip(particles[:, 0], alpha_bounds[0], alpha_bounds[1])
        particles[:, 1] = np.clip(particles[:, 1], beta_bounds[0], beta_bounds[1])

        # Evaluate all particles
        scores = objective_function(particles)

        # Update personal bests
        better_scores_mask = scores < personal_best_scores
        personal_best[better_scores_mask] = particles[better_scores_mask]
        personal_best_scores[better_scores_mask] = scores[better_scores_mask]

        # Update global best
        best_particle_idx = np.argmin(personal_best_scores)
        if personal_best_scores[best_particle_idx] < global_best_score:
            global_best = personal_best[best_particle_idx]
            global_best_score = personal_best_scores[best_particle_idx]

        # Early stopping
        if t > 0 and np.abs(global_best_score - personal_best_scores[best_particle_idx]) < early_stop_threshold:
            break

    # Apply the best parameters to enhance the image
    alpha, beta = global_best
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Apply pixel value clipping after enhancement to avoid brightness issues
    enhanced_image = np.clip(enhanced_image, 0, 255)

    print(f"Best alpha: {alpha:.3f}, Best beta: {beta:.3f}, Final Score: {-global_best_score:.6f}")
    return enhanced_image



# Metrics Calculation

def calculate_entropy(image):
    """Calculate the Shannon entropy of an image."""
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    histogram = histogram / histogram.sum()  # Normalize the histogram
    histogram = histogram[histogram > 0]  # Remove zero entries
    return -np.sum(histogram * np.log2(histogram))

def calculate_psnr(original, enhanced):
    """Calculate PSNR between two images."""
    # Compute MSE (Mean Squared Error)
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')  # If MSE is zero, PSNR is infinity (perfect match)
    max_pixel = 255.0  # For 8-bit images, the maximum pixel value is 255
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr_value

def calculate_metrics(original, enhanced):
    """Calculate PSNR, Entropy, and SSIM for two images."""
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Calculate PSNR
    psnr_value = calculate_psnr(original_gray, enhanced_gray)

    # Calculate Entropy
    entropy_value = calculate_entropy(enhanced_gray)

    # Calculate SSIM
    ssim_value = ssim(original_gray, enhanced_gray)

    return psnr_value, entropy_value, ssim_value


# Load image
image = cv2.imread('turtle.jpg')  # Replace with your image path

# Apply methods
fourier_enhanced = resize_to_original(fourier_enhancement(image), image)
wavelet_enhanced = resize_to_original(wavelet_enhancement(image), image)
enhanced_hsv, enhanced_ycbcr, enhanced_lab = map(lambda x: resize_to_original(x, image),
                                                 color_transform_enhancement(image))

pso_enhanced = resize_to_original(pso_enhancement(image), image)

# Enhanced images and method names
enhanced_images = [fourier_enhanced, wavelet_enhanced, enhanced_hsv, enhanced_ycbcr, enhanced_lab, pso_enhanced]
methods = ["Fourier", "Wavelet", "HSV", "YCbCr", "Lab", "PSO"]

# Compare images
display_comparisons(image, enhanced_images, methods)

# Compare histograms
plot_histogram_comparison(image, enhanced_images, methods)

# Calculate and display metrics
metrics = [calculate_metrics(image, enhanced) for enhanced in enhanced_images]
metrics_table = pd.DataFrame(metrics, columns=["PSNR", "Entropy", "SSIM"], index=methods)
print(metrics_table)