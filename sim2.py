import numpy as np
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beam.s4_beam import S4Beam
from shadow4.beamline.optical_elements.mirrors.s4_mirror import S4Mirror, S4MirrorElement
from shadow4.beamline.optical_elements.ideal_elements.s4_screen import S4Screen, S4ScreenElement
from shadow4.beamline.s4_beamline import S4Beamline
from syned.beamline.shape import Rectangle

def simulate_bimorph_beamline(voltages):
    source = SourceGeometrical(
        name='GeometricSource',
        spatial_type='Point',
        angular_distribution='Flat',
        energy_distribution='Single Line',
        nrays=50000,
        seed=12345
    )
    beam = S4Beam.create_from_source(source)

    mirror = S4Mirror(
        name="BimorphMirror",
        boundary_shape=Rectangle(x_left=-0.05, x_right=0.05, y_bottom=-0.25, y_top=0.25),
        f_reflec=1,  # 1 for full reflectivity
        f_refl=0,    # 0 for pre-calculated reflectivity (not used here)
    )

    x = np.linspace(-0.05, 0.05, 100)
    y = np.linspace(-0.25, 0.25, 500)
    X, Y = np.meshgrid(x, y)

    deformation = np.zeros_like(X)
    for i in range(8):
        norm_voltage = voltages[i] / 300.0
        deformation += norm_voltage * np.polynomial.legendre.legval(Y.flatten(), np.eye(8)[i]).reshape(X.shape) * 1e-6 # in meters


    with open("bimorph_surface.dat", "w") as f:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]} {Y[i, j]} {deformation[i, j]}\n")

    mirror.set_surface_from_file("bimorph_surface.dat")

    mirror_element = S4MirrorElement(
        optical_element=mirror,
        coordinates=None # Places the mirror at the origin
    )


    # 4. Define the Detector
    screen = S4Screen()
    screen_element = S4ScreenElement(
        optical_element=screen,
        coordinates=None # Places the screen after the mirror
    )

    # 5. Create the Beamline and Run the Simulation
    beamline = S4Beamline(
        light_source=source,
        beamline_elements=[mirror_element, screen_element]
    )

    # Trace the beam through the beamline
    output_beam, _ = beamline.trace_beam()

    # 6. Get the Detector Image
    # The ticket is a dictionary containing the histogram data
    ticket = output_beam.get_ticket()
    detector_image = ticket['histogram']

    return detector_image

if __name__ == '__main__':
    # Example usage:
    # Generate a random set of 8 voltages between -300V and 300V
    example_voltages = np.random.uniform(-300, 300, 8)

    # Run the simulation
    simulated_image = simulate_bimorph_beamline(example_voltages)

    # You can then visualize the image using a library like matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(simulated_image, cmap='hot', aspect='auto')
    plt.title("Simulated Detector Image")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.colorbar(label="Intensity")
    plt.show()
