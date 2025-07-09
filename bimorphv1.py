import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import threading
import time
import scipy.ndimage

# Shadow4 imports
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror
from shadow4.beamline.s4_beamline import S4Beamline
from shadow4.beam.s4_beam import S4Beam

class BimorphMirrorBeamline:
    def __init__(self):
        # Initialize voltage channels (-300V to +300V)
        self.voltages = np.zeros(8)
        self.voltage_range = (-300, 300)
        
        # Beamline parameters
        self.source_distance = 30.0  # meters
        self.mirror_distance = 50.0  # meters
        self.detector_distance = 2.0  # meters after mirror
        
        # Mirror parameters for horizontal focusing
        self.mirror_length = 0.5  # meters
        self.mirror_width = 0.05   # meters
        self.grazing_angle = 3.0   # mrad
        
        # Initialize beamline components
        self.setup_beamline()
        
        # Image parameters
        self.image_size = (256, 256)
        self.current_image = np.zeros(self.image_size)
        
        # Base mirror shape for deformation
        self.base_curvature = self.calculate_base_curvature()
        
    def setup_beamline(self):
        """Setup the Shadow4 beamline with source and bimorph mirror"""
        
        # Create X-ray source with correct parameters
        self.source = SourceGeometrical(
            spatial_type="Rectangle",
            angular_distribution="Flat",
            energy_distribution="Single Energy",
            photon_energy_ev=10000,  # 10 keV
            x_half_range=0.001,      # 1mm
            y_half_range=0.001,      # 1mm
            angular_x_half_range=0.0001,  # 0.1 mrad
            angular_y_half_range=0.0001,  # 0.1 mrad
            number_of_rays=5000
        )
        
        # Calculate ellipsoid parameters for horizontal line focus
        p = self.source_distance
        q = self.detector_distance
        theta = np.radians(self.grazing_angle / 1000)  # Convert mrad to rad
        
        # Create bimorph mirror (ellipsoid for horizontal focusing)
        self.mirror = S4EllipsoidMirror(
            name="Bimorph Mirror",
            boundary_shape="Rectangle",
            x_half_range=self.mirror_length/2,
            y_half_range=self.mirror_width/2,
            p_focus=p,
            q_focus=q,
            grazing_angle=theta,
            cylindrical=1,  # Cylindrical focusing (horizontal line focus)
            cylinder_direction=0  # 0 = tangential (horizontal)
        )
        
        # Create beamline
        self.beamline = S4Beamline()
        self.beamline.append_beamline_element(self.mirror)
        
    def calculate_base_curvature(self):
        """Calculate the base curvature profile for the ellipsoid"""
        # For horizontal line focus, we need the curvature in the sagittal direction
        x_positions = np.linspace(-self.mirror_length/2, self.mirror_length/2, 100)
        
        # Ellipsoid curvature for line focus
        p = self.source_distance
        q = self.detector_distance
        
        # Base curvature (simplified ellipsoid approximation)
        base_curvature = 2 / (p + q) * np.ones_like(x_positions)
        
        return base_curvature
        
    def apply_bimorph_deformation(self):
        """Apply bimorph deformation based on voltage channels"""
        
        # Create voltage-to-curvature mapping
        x_channels = np.linspace(-self.mirror_length/2, self.mirror_length/2, 8)
        x_mirror = np.linspace(-self.mirror_length/2, self.mirror_length/2, 100)
        
        # Voltage to curvature conversion (typical: ~1e-6 m^-1 per volt)
        curvature_coefficient = 1e-6  # 1/meters per volt
        
        # Calculate curvature changes for each channel
        curvature_changes = self.voltages * curvature_coefficient
        
        # Interpolate curvature changes across mirror surface
        curvature_profile = np.interp(x_mirror, x_channels, curvature_changes)
        
        # Apply to mirror (this is a simplified approach)
        # In reality, this would modify the mirror's conic constant or polynomial coefficients
        total_curvature = self.base_curvature + curvature_profile
        
        # Store the deformation for later use in image generation
        self.current_deformation = curvature_profile
        
    def generate_beam(self):
        """Generate beam and trace through beamline"""
        
        # Generate source beam
        beam = self.source.get_beam()
        
        # Apply bimorph deformation
        self.apply_bimorph_deformation()
        
        # Trace beam through beamline
        beam_out = self.beamline.trace_beam(beam)
        
        return beam_out
    
    def beam_to_image(self, beam):
        """Convert Shadow4 beam to 2D image array"""
        
        try:
            # Get beam coordinates at detector
            x = beam.get_column(1)  # Horizontal position
            y = beam.get_column(3)  # Vertical position
            intensity = np.ones(len(x))  # Uniform intensity for simplicity
            
            # Remove lost rays
            good_rays = beam.get_column(10) == 1
            x = x[good_rays]
            y = y[good_rays]
            intensity = intensity[good_rays]
            
            if len(x) == 0:
                return np.zeros(self.image_size)
            
            # Define detector area
            detector_size = 0.01  # 1cm detector
            x_range = [-detector_size/2, detector_size/2]
            y_range = [-detector_size/2, detector_size/2]
            
            # Create 2D histogram
            image, _, _ = np.histogram2d(
                y, x, 
                bins=self.image_size,
                range=[y_range, x_range],
                weights=intensity
            )
            
            # Apply voltage-dependent aberrations
            image = self.apply_voltage_effects(image)
            
            # Apply Gaussian smoothing for realistic detector response
            image = scipy.ndimage.gaussian_filter(image, sigma=1.0)
            
            return image
            
        except Exception as e:
            print(f"Error in beam_to_image: {e}")
            return self.generate_synthetic_image()
    
    def apply_voltage_effects(self, image):
        """Apply voltage-dependent effects to the image"""
        
        # Create effects based on voltage settings
        height, width = image.shape
        
        # Apply horizontal focusing/defocusing effects
        for i, voltage in enumerate(self.voltages):
            # Each channel affects a horizontal strip
            strip_width = width // 8
            x_start = i * strip_width
            x_end = (i + 1) * strip_width if i < 7 else width
            
            # Voltage affects the beam profile in that region
            effect_strength = voltage / 300.0  # Normalize to [-1, 1]
            
            # Apply focusing effect (compress/expand horizontally)
            if abs(effect_strength) > 0.01:
                for y in range(height):
                    strip = image[y, x_start:x_end]
                    if np.sum(strip) > 0:
                        # Simple focusing effect
                        center = len(strip) // 2
                        if effect_strength > 0:  # Focusing
                            # Compress toward center
                            compressed = np.zeros_like(strip)
                            compression_factor = 1 - abs(effect_strength) * 0.3
                            for x in range(len(strip)):
                                new_x = int(center + (x - center) * compression_factor)
                                if 0 <= new_x < len(strip):
                                    compressed[new_x] += strip[x]
                            image[y, x_start:x_end] = compressed
                        else:  # Defocusing
                            # Expand from center
                            expanded = np.zeros_like(strip)
                            expansion_factor = 1 + abs(effect_strength) * 0.3
                            for x in range(len(strip)):
                                new_x = int(center + (x - center) * expansion_factor)
                                if 0 <= new_x < len(strip):
                                    expanded[new_x] += strip[x] * 0.8  # Reduce intensity
                            image[y, x_start:x_end] = expanded
        
        return image
    
    def generate_synthetic_image(self):
        """Generate synthetic beam image when Shadow4 fails"""
        
        # Create a base horizontal line focus
        image = np.zeros(self.image_size)
        center_y = self.image_size[0] // 2
        
        # Create horizontal line with Gaussian profile in vertical direction
        y_indices = np.arange(self.image_size[0])
        vertical_profile = np.exp(-((y_indices - center_y) ** 2) / (2 * 10**2))
        
        # Create base horizontal line
        for x in range(self.image_size[1]):
            image[:, x] = vertical_profile
        
        # Apply voltage effects
        image = self.apply_voltage_effects(image)
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.1, image.shape)
        image = np.maximum(0, image + noise)
        
        return image
    
    def update_image(self):
        """Update the beam image based on current voltages"""
        try:
            # Try to use Shadow4
            beam = self.generate_beam()
            self.current_image = self.beam_to_image(beam)
            return True
        except Exception as e:
            print(f"Shadow4 error, using synthetic image: {e}")
            # Fall back to synthetic image
            self.current_image = self.generate_synthetic_image()
            return True
    
    def set_voltage(self, channel, voltage):
        """Set voltage for a specific channel (0-7)"""
        if 0 <= channel <= 7:
            self.voltages[channel] = np.clip(voltage, self.voltage_range[0], self.voltage_range[1])
            return True
        return False
    
    def get_image(self):
        """Get current beam image as numpy array"""
        return self.current_image.copy()
    
    def reset_voltages(self):
        """Reset all voltages to zero"""
        self.voltages = np.zeros(8)

class RealtimeBeamlineGUI:
    def __init__(self):
        self.beamline = BimorphMirrorBeamline()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup matplotlib GUI with sliders"""
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 8))
        
        # Image display
        self.ax_image = plt.subplot(1, 2, 1)
        self.ax_image.set_title('Beam Profile at Detector (Horizontal Line Focus)')
        self.ax_image.set_xlabel('Horizontal Position (mm)')
        self.ax_image.set_ylabel('Vertical Position (mm)')
        
        # Initialize image display
        self.beamline.update_image()
        self.im = self.ax_image.imshow(
            self.beamline.get_image(),
            extent=[-5, 5, -5, 5],  # 1cm detector in mm
            origin='lower',
            cmap='hot',
            interpolation='bilinear'
        )
        plt.colorbar(self.im, ax=self.ax_image, label='Intensity')
        
        # Voltage controls
        self.ax_controls = plt.subplot(1, 2, 2)
        self.ax_controls.set_title('Bimorph Mirror Voltage Controls')
        self.ax_controls.axis('off')
        
        # Create sliders
        self.sliders = []
        slider_height = 0.03
        slider_spacing = 0.08
        
        for i in range(8):
            ax_slider = plt.axes([0.6, 0.8 - i*slider_spacing, 0.3, slider_height])
            slider = Slider(
                ax_slider,
                f'Ch {i+1}',
                self.beamline.voltage_range[0],
                self.beamline.voltage_range[1],
                valinit=0,
                valfmt='%0.0f V'
            )
            slider.on_changed(lambda val, ch=i: self.update_voltage(ch, val))
            self.sliders.append(slider)
        
        # Add preset buttons
        self.add_preset_buttons()
        
        # Add voltage display
        self.add_voltage_display()
        
        # Start update timer
        self.timer = self.fig.canvas.new_timer(interval=200)  # 200ms updates
        self.timer.add_callback(self.update_display)
        self.timer.start()
        
    def add_preset_buttons(self):
        """Add preset voltage configuration buttons"""
        
        # Flat mirror preset
        ax_flat = plt.axes([0.6, 0.15, 0.08, 0.04])
        btn_flat = Button(ax_flat, 'Flat')
        btn_flat.on_clicked(lambda x: self.apply_preset([0]*8))
        
        # Focusing preset
        ax_focus = plt.axes([0.7, 0.15, 0.08, 0.04])
        btn_focus = Button(ax_focus, 'Focus')
        btn_focus.on_clicked(lambda x: self.apply_preset([50, 100, 150, 200, 200, 150, 100, 50]))
        
        # Defocusing preset
        ax_defocus = plt.axes([0.8, 0.15, 0.08, 0.04])
        btn_defocus = Button(ax_defocus, 'Defocus')
        btn_defocus.on_clicked(lambda x: self.apply_preset([-50, -100, -150, -200, -200, -150, -100, -50]))
        
        # Aberration correction preset
        ax_aberr = plt.axes([0.6, 0.1, 0.08, 0.04])
        btn_aberr = Button(ax_aberr, 'Correct')
        btn_aberr.on_clicked(lambda x: self.apply_preset([30, -20, 80, -40, -40, 80, -20, 30]))
        
        # Reset button
        ax_reset = plt.axes([0.8, 0.1, 0.08, 0.04])
        btn_reset = Button(ax_reset, 'Reset')
        btn_reset.on_clicked(lambda x: self.apply_preset([0]*8))
        
    def add_voltage_display(self):
        """Add voltage value display"""
        self.voltage_text = self.ax_controls.text(0.6, 0.05, '', transform=self.ax_controls.transAxes)
        
    def apply_preset(self, voltages):
        """Apply preset voltage configuration"""
        for i, voltage in enumerate(voltages):
            self.sliders[i].set_val(voltage)
            self.beamline.set_voltage(i, voltage)
        
    def update_voltage(self, channel, voltage):
        """Update voltage for specific channel"""
        self.beamline.set_voltage(channel, voltage)
        
    def update_display(self):
        """Update the beam image display"""
        if self.beamline.update_image():
            image = self.beamline.get_image()
            self.im.set_array(image)
            if np.max(image) > 0:
                self.im.set_clim(vmin=0, vmax=np.max(image))
            
            # Update voltage display
            voltage_str = "Voltages: " + ", ".join([f"{v:.0f}V" for v in self.beamline.voltages])
            self.voltage_text.set_text(voltage_str)
            
            self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the GUI"""
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the beamline simulator"""
    
    print("Starting Shadow4 Bimorph Mirror Beamline Simulator...")
    print("Features:")
    print("- 8 voltage channels (-300V to +300V)")
    print("- Real-time beam profile updates")
    print("- Horizontal line focus configuration")
    print("- Preset voltage configurations")
    print("- Fallback to synthetic simulation if Shadow4 fails")
    print("\nAdjust the voltage sliders to see real-time changes in the beam profile.")
    
    # Create and run GUI
    gui = RealtimeBeamlineGUI()
    gui.show()

if __name__ == "__main__":
    main()
