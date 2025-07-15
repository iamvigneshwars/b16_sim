import numpy as np
import matplotlib.pyplot as plt

def simulate_detector_image(voltages: np.ndarray) -> np.ndarray:
    if voltages.shape != (8,):
        raise ValueError("Voltages must be a NumPy array of exactly 8 elements.")
    
    optimal_voltages = np.array([248.0, 156.0, -134.0, 90.0, 43.0, 222.0, -14.0, -64.0])
    
    dev = np.linalg.norm(voltages - optimal_voltages)
    
    min_sigma = 21.7442  # Minimum vertical spread (pixels) from example
    sigma = min_sigma + dev / 20.0  # Increases slowly with deviation
    max_amp = 4133.65  # Maximum peak amplitude from example
    amp = max_amp * np.exp(- (dev / 200.0)**2)  # Intensity decreases very slowly with deviation
    offset = 482.643  # Background offset from example
    center = 91.6337  # Center from example
    
    height, width = 180, 50
    
    y = np.arange(height)
    vertical_profile = offset + amp * np.exp(-0.5 * ((y - center) / sigma)**2)
    
    image = np.tile(vertical_profile[:, np.newaxis], (1, width))
    
    # Clip and cast to uint16
    image = np.clip(image, 0, 65535).astype(np.uint16)
    
    return image


if __name__ == "__main__":
    voltages_set = np.random.uniform(-300, 300, size=(500, 8))

    plt.ion()
    fig, ax = plt.subplots()
    im = None

    for i, voltages in enumerate(voltages_set):
        image = simulate_detector_image(voltages)
        if im is None:
            im = ax.imshow(image, cmap='gray', aspect='auto')
            plt.colorbar(im, ax=ax, label='Intensity')
            plt.xlabel('Horizontal Pixel')
            plt.ylabel('Vertical Pixel')
        else:
            im.set_data(image)
            im.set_clim(vmin=image.min(), vmax=image.max())

        ax.set_title(f'Simulated Detector Image (Set {i+1}/{len(voltages_set)})')
        plt.draw()
        plt.pause(0.1)

    plt.ioff()

