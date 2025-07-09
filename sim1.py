import numpy as np
import matplotlib.pyplot as plt
from shadow4.beam.s4_beam import S4Beam
from shadow4.beamline.optical_elements.mirrors.s4_numerical_mesh_mirror import S4NumericalMeshMirror, S4NumericalMeshMirrorElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.shape import Rectangle
from scipy.interpolate import interp1d
from shadow4.sources.source_geometrical.source_gaussian import SourceGaussian

def simulate_beamline(voltages=None, L=100e-3, W=10e-3, N_x=100, N_y=50, k=1e-9):
    # Create 1D coordinate arrays
    xx = np.linspace(0, L, N_x)
    yy = np.linspace(-W/2, W/2, N_y)
    
    # Calculate parabolic surface for focusing
    p = 20.0  # source-to-mirror distance (m)
    q = 1.0   # mirror-to-detector distance (m)
    theta = np.radians(2.0)  # grazing angle
    f = 2 / (1/p + 1/q) / np.sin(theta)  # focal length
    z_ideal = xx**2 / (4 * f * np.sin(theta))  # parabolic profile
    
    # If voltages provided, scale to match ideal profile
    if voltages is not None:
        n_act = len(voltages)
        x_act = np.linspace(0, L, n_act+2)[1:-1]
        z_act = k * voltages
        z_func = interp1d(x_act, z_act, kind='cubic', fill_value=0.0, bounds_error=False)
        z_xx = z_func(xx)
    else:
        z_xx = z_ideal  # Use ideal parabolic profile
    
    # Create 2D height array
    zz = np.zeros((N_x, N_y))
    for i in range(N_x):
        zz[i, :] = z_xx[i]
    zz = zz.T
    
    # Define mirror and beamline elements
    boundary_shape = Rectangle(x_left=0, x_right=L, y_bottom=-W/2, y_top=W/2)
    mirror = S4NumericalMeshMirror(name="bimorph", boundary_shape=boundary_shape, xx=xx, yy=yy, zz=zz)
    coordinates = ElementCoordinates(p=20000e-3, q=1000e-3, angle_radial=88.0, angle_azimuthal=0.0)
    
    # Optimized Gaussian source
    source = SourceGaussian.initialize_from_keywords(
        nrays=10000,
        sigmaX=50e-6, sigmaZ=50e-6,  # 50 μm source size
        sigmaXprime=50e-6, sigmaZprime=50e-6  # 50 μrad divergence
    )
    beam = source.get_beam()
    
    # Create optical element
    element = S4NumericalMeshMirrorElement(
        optical_element=mirror,
        coordinates=coordinates,
        movements=S4BeamlineElementMovements(),
        input_beam=beam
    )
    
    # Trace the beam
    output_beam, _ = element.trace_beam()
    
    # Extract good rays and intensity
    good_rays = output_beam.rays_good
    x = good_rays[:, 0]  # X (column 1)
    z = good_rays[:, 2]  # Z (column 3)
    intensity = output_beam.get_column(23, nolost=1)  # Total intensity
    
    # Create histogram with intensity weighting
    xbins = np.linspace(min(x), max(x), 101)
    zbins = np.linspace(min(z), max(z), 201)  # Finer bins in Z for focus
    image, _, _ = np.histogram2d(x, z, bins=(xbins, zbins), weights=intensity)
    
    return image, xbins, zbins

def visualize_image(image, xbins, zbins):
    plt.figure(figsize=(8, 6))
    plt.imshow(image.T, origin='lower', extent=[xbins[0], xbins[-1], zbins[0], zbins[-1]],
               aspect='auto', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('Detector Image (Line Focus)')
    plt.show()

if __name__ == "__main__":
    # Test with ideal parabolic profile (no voltages)
    image, xbins, zbins = simulate_beamline(voltages=None)
    print("Simulation completed. Image shape:", image.shape)
    visualize_image(image, xbins, zbins)
    
    # Test with user-defined voltages
    voltages = np.array([0, 50, -50, 100, -100, 50, -50, 0])
    image, xbins, zbins = simulate_beamline(voltages)
    print("Simulation completed with voltages. Image shape:", image.shape)
    visualize_image(image, xbins, zbins)
