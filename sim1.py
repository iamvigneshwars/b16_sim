import numpy as np
from shadow4.beam.s4_beam import S4Beam
from shadow4.beamline.optical_elements.mirrors.s4_numerical_mesh_mirror import S4NumericalMeshMirror, S4NumericalMeshMirrorElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.shape import Rectangle
from scipy.interpolate import interp1d
from shadow4.sources.source_geometrical.source_gaussian import SourceGaussian

def simulate_beamline(voltages, L=100e-3, W=10e-3, N_x=100, N_y=50, k=1e-9):
    # Create 1D coordinate arrays
    xx = np.linspace(0, L, N_x)
    yy = np.linspace(-W/2, W/2, N_y)
    
    # Calculate deformation along x-direction
    n_act = len(voltages)
    x_act = np.linspace(0, L, n_act+2)[1:-1]
    z_act = k * voltages
    z_func = interp1d(x_act, z_act, kind='cubic', fill_value=0.0, bounds_error=False)
    
    # Create 2D height array: zz[i,j] corresponds to xx[i], yy[j]
    zz = np.zeros((N_x, N_y))
    for i in range(N_x):
        zz[i, :] = z_func(xx[i])
    zz = zz.T
    
    boundary_shape = Rectangle(x_left=0, x_right=L, y_bottom=-W/2, y_top=W/2)
    mirror = S4NumericalMeshMirror(name="bimorph", boundary_shape=boundary_shape, 
                                  xx=xx, yy=yy, zz=zz)
    coordinates = ElementCoordinates(p=20000e-3, q=1000e-3, angle_radial=88.0, angle_azimuthal=0.0)
    source = SourceGaussian.initialize_from_keywords(nrays=10000, sigmaX=1e-3, sigmaZ=1e-3)
    beam = source.get_beam()
    element = S4NumericalMeshMirrorElement(
        optical_element=mirror,
        coordinates=coordinates,
        movements=S4BeamlineElementMovements(),
        input_beam=beam
    )
    output_beam, _ = element.trace_beam()
    good = output_beam.get_good_rays()
    x = output_beam.get_column(1, good)
    y = output_beam.get_column(3, good)
    xbins = np.linspace(min(x), max(x), 101)
    ybins = np.linspace(min(y), max(y), 101)
    image, _, _ = np.histogram2d(x, y, bins=(xbins, ybins))
    return image

if __name__ == "__main__":
    voltages = np.array([0, 50, -50, 100, -100, 50, -50, 0])
    image = simulate_beamline(voltages)
    print("Simulation completed. Image shape:", image.shape)
