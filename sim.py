import numpy as np
import matplotlib.pyplot as plt

# Shadow4 and Syned imports
from shadow4.beam.s4_beam import S4Beam
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.mirrors.s4_additional_numerical_mesh_mirror import S4AdditionalNumericalMeshMirror
from shadow4.beamline.optical_elements.mirrors.s4_numerical_mesh_mirror import S4NumericalMeshMirror
from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror
from shadow4.beamline.s4_beamline import S4Beamline
from shadow4.beamline.s4_beamline_element import S4BeamlineElement
from syned.beamline.element_coordinates import ElementCoordinates

def run_beamline_simulation(voltages):
    """
    This function simulates a simple beamline with a bimorph mirror.
    ...
    """
    if voltages.shape != (8,) or np.any(voltages < -300) or np.any(voltages > 300):
        raise ValueError("Voltages must be a 1D NumPy array of 8 values between -300V and 300V.")

    # 1. Define the Light Source
    light_source = SourceGeometrical()
    light_source.set_energy_distribution_singleline(value=1000, unit='eV')
    light_source.set_spatial_type_gaussian(sigma_h=1e-4, sigma_v=1e-4)
    light_source.set_angular_distribution_gaussian(sigdix=1e-4, sigdiz=1e-4)
    light_source.set_nrays(100000)

    # 2. Define the Bimorph Mirror
    base_mirror = S4PlaneMirror(name="Base Plane Mirror", f_reflec=1, f_refl=0)

    x = np.linspace(-0.05, 0.05, 100)
    y = np.linspace(-0.2, 0.2, 200)
    X, Y = np.meshgrid(x, y)
    
    z = np.zeros_like(X)
    for i in range(8):
        z += voltages[i] * 1e-9 * np.sin((i + 1) * np.pi * (Y / 0.2))

    mesh_object = S4NumericalMeshMirror(xx=X.flatten(), yy=Y.flatten(), zz=z.flatten())
    mirror = S4AdditionalNumericalMeshMirror(name="Bimorph Mirror", ideal_mirror=base_mirror, numerical_mesh_mirror=mesh_object)

    # 3. Define the Beamline
    beamline = S4Beamline()
    beamline.set_light_source(light_source)
    coordinates = ElementCoordinates(p=10.0, q=10.0, angle_radial=np.deg2rad(89.8), angle_azimuthal=0.0)
    
    # We append the optical element and coordinates. The beamline now serves as a simple container for our components.
    beamline.append_beamline_element(S4BeamlineElement(optical_element=mirror, coordinates=coordinates))

    # 4. Run the Simulation
    # ============================ MODIFICATION START ============================
    # Get the initial beam from the source
    beam = light_source.get_beam()

    # Loop through the components stored in the beamline object
    for element_definition in beamline.get_beamline_elements():
        # Create a new S4BeamlineElement, passing the BEAM to the constructor
        tracer_element = S4BeamlineElement(
            optical_element=element_definition.get_optical_element(),
            coordinates=element_definition.get_coordinates(),
            input_beam=beam
        )
        
        # Call trace_beam() with NO arguments. It uses the input_beam provided in the constructor.
        beam, _ = tracer_element.trace_beam()

    final_beam = beam
    # ============================= MODIFICATION END =============================

    # 5. Get the Detector Image
    detector_image, _, _ = final_beam.get_histogram(nbins=256)

    return detector_image

if __name__ == '__main__':
    voltages = np.array([100, -50, 200, -150, 50, -250, 300, -100])
    detector_image = run_beamline_simulation(voltages)

    plt.figure(figsize=(8, 6))
    plt.imshow(detector_image, cmap='viridis', origin='lower')
    plt.title("Simulated Detector Image")
    plt.xlabel("X [pixels]")
    plt.ylabel("Y [pixels]")
    plt.colorbar(label="Intensity")
    plt.show()
