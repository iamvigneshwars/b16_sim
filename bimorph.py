import numpy as np
import os
from PIL import Image

# Required imports for the SHADOW simulation
from shadow4.beamline.s4_beamline import S4Beamline
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.mirrors.s4_mirror import S4Mirror, S4MirrorElement
from syned.beamline.shape import Rectangle
# --- ADD THIS IMPORT for specifying surface calculation method ---
from shadow4.optical_elements.s4_optical_element import SurfaceCalculation

print("Successfully imported shadow4 and Pillow libraries.")

class BimorphMirrorSimulator:
    """
    A class to simulate a bimorph mirror beamline using SHADOW.
    """
    def __init__(self, config: dict):
        """
        Initializes the simulator with beamline and mirror parameters.
        """
        print("Initializing Bimorph Mirror Simulator...")
        self.config = config
        
        self.mirror_length = config['mirror_length']
        self.mirror_width = config['mirror_width']
        self.actuator_positions_y = np.linspace(-self.mirror_length / 2.1, self.mirror_length / 2.1, 8)
        
        # The beamline is now initialized with only the static components.
        self._initialize_beamline()
        print("Simulator initialized.")

    def _initialize_beamline(self):
        """Creates the static SHADOW source. The mirror is now created dynamically."""
        self.light_source = SourceGeometrical(
            name='GeometricSource',
            nrays=self.config['n_rays'],
            seed=0 # for reproducibility
        )

    def _calculate_mirror_surface_from_voltages(self, voltages: list) -> str:
        """
        Calculates a 2D surface height map from 8 actuator voltages.
        (This method remains unchanged)
        """
        print(f"  Calculating mirror surface for voltages: {np.round(voltages, 2)}")
        
        x_coords = np.linspace(-self.mirror_width / 2, self.mirror_width / 2, 101)
        y_coords = np.linspace(-self.mirror_length / 2, self.mirror_length / 2, 501)
        total_surface_z = np.zeros((len(x_coords), len(y_coords)))
        actuator_sigma = self.config.get('actuator_influence_width', self.mirror_length / 10)
        voltage_to_height_nm = self.config.get('voltage_to_height_nm', 0.4)

        for i in range(8):
            actuator_pos_y = self.actuator_positions_y[i]
            gaussian_influence_y = np.exp(-((y_coords - actuator_pos_y)**2) / (2 * actuator_sigma**2))
            actuator_height_m = (voltages[i] * voltage_to_height_nm) * 1e-9
            total_surface_z += actuator_height_m * gaussian_influence_y[np.newaxis, :]

        surface_file_path = "bimorph_surface_temp.dat"
        with open(surface_file_path, 'w') as f:
            for i in range(len(x_coords)):
                for j in range(len(y_coords)):
                    f.write(f"{x_coords[i]} {y_coords[j]} {total_surface_z[i, j]}\n")
        return surface_file_path

    def run_and_save_tiff(self, voltages: list, output_tiff_path: str):
        """
        Runs the full simulation for a given set of voltages and saves the
        resulting beam image as a TIFF file.
        """
        # 1. Calculate the surface file for the current run.
        surface_file = self._calculate_mirror_surface_from_voltages(voltages)

        # 2. Define the mirror's physical shape (aperture).
        boundary_shape = Rectangle(
            x_left=-self.config['mirror_width'] / 2,
            x_right=self.config['mirror_width'] / 2,
            y_bottom=-self.config['mirror_length'] / 2,
            y_top=self.config['mirror_length'] / 2,
        )

        # 3. Create the mirror object HERE, inside the run method.
        #    This ensures it is created with the correct surface file for this specific run.
        mirror_oe = S4Mirror(
            name="BimorphMirror",
            boundary_shape=boundary_shape,
            # Tell the mirror its surface comes from an external file
            surface_calculation=SurfaceCalculation.EXTERNAL,
            file_with_surface_data=surface_file,
            f_reflec=0, # 0=perfect reflectivity
        )
        
        # 4. Create the beamline element that places the mirror.
        mirror_element = S4MirrorElement(
            optical_element=mirror_oe,
            p=self.config['source_to_mirror_dist'], 
            q=self.config['mirror_to_focus_dist'], 
            grazing_angle=self.config['grazing_angle_mrad'] * 1e-3,
        )

        # 5. Assemble and run the beamline.
        print("  Assembling beamline and starting ray-tracing...")
        beamline = S4Beamline(
            light_source=self.light_source,
            beamline_elements_list=[mirror_element]
        )
        final_beam, _ = beamline.run_beamline()
        print("  Ray-tracing complete.")

        # 6. Generate histogram and save the TIFF file.
        print("  Generating histogram from final beam...")
        hist_data, _, _ = final_beam.histo2(
            1, 3, nbins=self.config['image_resolution_pixels'], ref=23
        )
        print(f"  Saving beam image to: {output_tiff_path}")
        if hist_data.max() > 0:
            image_data = (hist_data / hist_data.max() * 65535).astype(np.uint16)
        else:
            image_data = np.zeros_like(hist_data, dtype=np.uint16)
        image = Image.fromarray(image_data.T, mode='I;16')
        image.save(output_tiff_path)

        # 7. Clean up the temporary file.
        os.remove(surface_file)
        print("  Cleanup complete.")
