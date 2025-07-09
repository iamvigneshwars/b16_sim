import numpy as np
from shadow4.beam.s4_beam import S4Beam
from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror, S4EllipsoidMirrorElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.beamline.optical_elements.s4_empty import S4EmptyElement
from shadow4.sources.source_geometrical.source_gaussian import SourceGaussian

def create_beamline(voltages):
    """
    Create and trace a beamline with a Gaussian source, bimorph mirror, and detector screen.
    Args:
        voltages (list): 8 voltages (0-100 V) for bimorph mirror segments.
    Returns:
        np.ndarray: 2D histogram of beam footprint (200x200).
    """
    # Gaussian source (undulator-like)
    source = SourceGaussian(
        nrays=10000,  # Number of rays
        sigma_X=1e-3,  # 1 mm beam size
        sigma_Y=1e-3,
        sigma_Z=1e-3,
        sigma_div_x=1e-4,  # 0.1 mrad divergence
        sigma_div_y=1e-4
    )
    beam = source.get_beam()
    
    # Bimorph mirror (ellipsoidal)
    mirror = S4EllipsoidMirror(
        name="Bimorph Mirror",
        boundary_shape=None,
        surface_calculation=0,  # Internal calculation
        min_axis=0.1,  # Ellipse minor axis (m)
        maj_axis=0.5,  # Ellipse major axis (m)
        pole_to_focus=1.0,  # Source to mirror distance (m)
        p_focus=1.0,  # Source to mirror
        q_focus=1.0,  # Mirror to focus
        grazing_angle=0.003  # 3 mrad
    )
    
    # Apply slope errors based on voltages
    n_segments = 8
    segment_length = 0.8 / n_segments  # Mirror length 0.8 m
    slope_errors = [v * 1e-7 for v in voltages]  # 100 V -> 10 nrad slope error
    mirror.set_surface_error(
        error_type=1,  # Slope error
        error_profile=np.array(slope_errors),
        error_profile_x=np.linspace(-0.4, 0.4, n_segments)  # Mirror segments
    )
    
    mirror_element = S4EllipsoidMirrorElement(
        optical_element=mirror,
        coordinates=S4BeamlineElementMovements()
    )
    
    # Detector screen
    screen_element = S4EmptyElement(
        coordinates=S4BeamlineElementMovements(distance=1.0)  # 1 m from mirror
    )
    
    # Trace beam through mirror
    beam, _ = mirror_element.trace_beam(beam)
    
    # Trace to screen
    beam, _ = screen_element.trace_beam(beam)
    
    # Compute 2D histogram of beam footprint
    rays = beam.get_rays()
    x = rays[:, 0] * 1e3  # Convert to mm
    y = rays[:, 1] * 1e3
    hist, xedges, yedges = np.histogram2d(
        x, y, bins=(200, 200), range=((-5, 5), (-5, 5))
    )
    
    # Normalize to 0-255 for consistency
    hist = (hist / np.max(hist, initial=1) * 255).astype(np.uint8)
    return hist.T  # Transpose for correct orientation

def main():
    # Example voltage configurations
    voltage_sets = [
        [50.0] * 8,  # Uniform 50 V
        [100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0],  # Alternating
        [0.0, 25.0, 50.0, 75.0, 100.0, 75.0, 50.0, 25.0]  # Linear gradient
    ]
    
    # Simulate for each voltage set
    for i, voltages in enumerate(voltage_sets):
        print(f"\nVoltage Set {i + 1}: {voltages}")
        detector_image = create_beamline(voltages)
        print(f"Detector Image Shape: {detector_image.shape}")
        print(f"Sample Pixel Values (center 5x5):\n{detector_image[97:102, 97:102]}")
        # Optionally save or process the array further
        # e.g., np.save(f"detector_image_set_{i+1}.npy", detector_image)

if __name__ == "__main__":
    main()
