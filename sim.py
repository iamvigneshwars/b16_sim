import argparse
import numpy as np
from matplotlib import pyplot as plt
from mirror import simulate

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulate bimorph mirror and display detector image.')
    parser.add_argument('--voltages', nargs=8, type=float, metavar='V', help='A list of 8 voltages for a single simulation.')
    args = parser.parse_args()

    if args.voltages:
        voltages = args.voltages
        print(f"Simulating with voltages: {voltages}")
        image = simulate(voltages)

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(image, cmap='inferno', aspect='auto')
        plt.colorbar(im, ax=ax, label='Intensity')
        plt.xlabel('Horizontal Pixel')
        plt.ylabel('Vertical Pixel')
        ax.set_title(f'Simulated Detector Image\nVoltages: {voltages}')
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
                plt.xlabel('Horizontal Pixel')
                plt.ylabel('Vertical Pixel')
            else:
                im.set_data(image)
                im.set_clim(vmin=image.min(), vmax=image.max())

            ax.set_title(f'Simulated Detector Image (Set {i+1}/{len(voltages_set)})')
            plt.draw()
            plt.pause(0.1)

        plt.ioff()
        plt.show()
