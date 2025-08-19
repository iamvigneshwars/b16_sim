import argparse
import numpy as np
from matplotlib import pyplot as plt
from mirror import simulate
from processing import BeamAnalyzer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulate bimorph mirror and display detector image.')
    parser.add_argument('--voltages', nargs=8, type=float, metavar='V', help='A list of 8 voltages for a single simulation.')
    args = parser.parse_args()

    if args.voltages:
        voltages = args.voltages
        print(f"Simulating with voltages: {voltages}")
        image = simulate(voltages)
        analyzer = BeamAnalyzer(image=image, crop=False)
        beam_properties = analyzer.analyze(profile_width=10)
        print(f"Vertical Beam Size: {beam_properties.vertical_beam_size:.4f}")
        print(f"FWHM: {beam_properties.fwhm:.4f}")
        print(f"Peak Intensity: {beam_properties.peak_intensity:.2f}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        im = ax1.imshow(image, cmap='inferno', aspect='auto')
        plt.colorbar(im, ax=ax1, label='Intensity')
        ax1.set_title('Simulated Detector Image')
        ax1.set_xlabel('Horizontal Position')
        ax1.set_ylabel('Vertical Position')
        
        y_coords = np.arange(len(beam_properties.vertical_line_profile))
        ax2.plot(beam_properties.vertical_line_profile, y_coords)
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Vertical Position')
        ax2.set_title('Vertical Line Profile')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    else:
        voltages_set = np.random.uniform(-5, 5, size=(500, 8))
        plt.ion()
        fig, ax = plt.subplots(figsize=(14, 8))
        im = None

        for i, voltages in enumerate(voltages_set):
            image = simulate(voltages)
            if im is None:
                im = ax.imshow(image, cmap='inferno', aspect='auto')
                plt.colorbar(im, ax=ax, label='Intensity')
            else:
                im.set_data(image)
                im.set_clim(vmin=image.min(), vmax=image.max())

            ax.set_title(f'Simulated Detector Image (Set {i+1}/{len(voltages_set)})')
            plt.draw()
            plt.pause(0.1)

        plt.ioff()
        plt.show()
