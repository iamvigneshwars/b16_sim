import numpy as np
from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.beam.s4_beam import S4Beam
from shadow4.sources.source_geometrical.source_gaussian import SourceGaussian
from syned.beamline.shape import  Ellipse
from shadow4.beamline.optical_elements.mirrors.s4_conic_mirror import S4ConicMirror, S4ConicMirrorElement
from shadow4.beamline.optical_elements.absorbers.s4_screen import S4Screen, S4ScreenElement
from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror, S4EllipsoidMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_numerical_mesh_mirror import S4NumericalMeshMirror
from shadow4.beamline.optical_elements.mirrors.s4_additional_numerical_mesh_mirror import S4AdditionalNumericalMeshMirror
from shadow4.beamline.optical_elements.mirrors.s4_additional_numerical_mesh_mirror import S4AdditionalNumericalMeshMirrorElement

def calculate_bimorph_deformation(voltages, yy, alpha=1e-3):
    """
    Compute 1D height deformation z(y) from 8 voltages.
    
    Parameters:
    voltages (list or array): 8 voltages in [-300, 300] V.
    yy (np.ndarray): 1D array of Y-coordinates (m).
    alpha (float): Sensitivity in m⁻¹/V (adjust per mirror specs).
    
    Returns:
    np.ndarray: 1D array of z heights (m).
    """
    if len(voltages) != 8:
        raise ValueError("Exactly 8 voltages required.")
    ny = len(yy)
    dy = yy[1] - yy[0]
    segment_len = ny // 8
    curvature = np.zeros(ny)
    for i, v in enumerate(voltages):
        start = i * segment_len
        end = min((i + 1) * segment_len, ny)
        curvature[start:end] = alpha * v
    # Integrate curvature to slope
    slope = np.cumsum(curvature) * dy
    # Detrend slope (e.g., zero at ends for fixed boundaries)
    slope -= np.linspace(slope[0], slope[-1], ny)
    # Integrate slope to height
    z = np.cumsum(slope) * dy
    # Normalize to zero at center
    z -= z[ny // 2]
    return z

def simulate(voltages=[0.0] * 8):

    boundary_shape = Ellipse(a_axis_min=-0.000005, a_axis_max=0.000005, b_axis_min=-0.01, b_axis_max=0.01)
    source = SourceGaussian(nrays=500000,
                 sigmaX=0.0,
                 sigmaY=0.0,
                 sigmaZ=0.0,
                 sigmaXprime=1e-6,
                 sigmaZprime=1e-6,)
    beam0 = S4Beam()
    beam0.generate_source(source)
    print(beam0.info())

    xmin, xmax = -5e-6, 5e-6  # From your boundary_shape a_axis
    ymin, ymax = -0.01, 0.01  # From b_axis
    nx, ny = 51, 201
    xx = np.linspace(xmin, xmax, nx)
    yy = np.linspace(ymin, ymax, ny)

    z_deform = calculate_bimorph_deformation(voltages, yy, alpha=1e-3)

    zz = np.tile(z_deform[:, np.newaxis], (1, nx))

    ideal_mirror = S4EllipsoidMirror(
        name="Ideal Ellipsoid",
        boundary_shape=boundary_shape,
        p_focus=10.0,
        q_focus=6.0,
        grazing_angle=np.radians(1.2)
    )

    deform_mirror = S4NumericalMeshMirror(
        name="Bimorph Deformation",
        boundary_shape=boundary_shape,
        xx=xx,
        yy=yy,
        zz=zz
    )
    bimorph_mirror = S4AdditionalNumericalMeshMirror(
        ideal_mirror=ideal_mirror,
        numerical_mesh_mirror=deform_mirror,
        name="Bimorph Mirror"
    )

    coordinates_syned = ElementCoordinates(p = 10.0,
                                           q = 6.0,
                                           angle_radial = np.radians(88.8))

    mirror_element = S4AdditionalNumericalMeshMirrorElement(
        optical_element=bimorph_mirror,
        coordinates=coordinates_syned,
        input_beam=beam0
    )

    beam1, _ = mirror_element.trace_beam()


    s1 = S4ScreenElement(optical_element=S4Screen(),
                         coordinates=ElementCoordinates(p=0.0, q=1.0, angle_radial=0.0),
                         input_beam=beam1)

    beam_on_source, _ = s1.trace_beam()

    rays = beam_on_source.rays_good
    vertical_size = np.std(rays[:, 2]) * 1e6  # Vertical RMS in μm (adjust column if needed)
    intensity_fraction = len(rays) / 500000.0
    print(f"Vertical size: {vertical_size:.2f} μm, Intensity fraction: {intensity_fraction:.4f}")
    x = rays[:, 0]
    z = rays[:, 2]
    
    image_data, _, _ = np.histogram2d(x, z, bins=201, range=[[-100e-6, 100e-6], [-100e-6, 100e-6]])
    return image_data

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    voltages_set = np.random.uniform(-300, 300, size=(500, 8))

    plt.ion()
    fig, ax = plt.subplots()
    im = None

    for i, voltages in enumerate(voltages_set):
        image = simulate(voltages)
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
