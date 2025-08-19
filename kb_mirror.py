import numpy
from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.beam.s4_beam import S4Beam
from shadow4.sources.source_geometrical.source_gaussian import SourceGaussian
from syned.beamline.shape import Direction, Side
from shadow4.beamline.s4_optical_element_decorators import SurfaceCalculation
from shadow4.beamline.optical_elements.mirrors.s4_conic_mirror import S4ConicMirror, S4ConicMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_toroid_mirror import S4ToroidMirror, S4ToroidMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror, S4PlaneMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror, S4EllipsoidMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_hyperboloid_mirror import S4HyperboloidMirror, S4HyperboloidMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_paraboloid_mirror import S4ParaboloidMirror, S4ParaboloidMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_sphere_mirror import S4SphereMirror, S4SphereMirrorElement
from shadow4.tools.graphics import plotxy

def example_branch_2(do_plot=True):
    source = SourceGaussian(nrays=10000,
                 sigmaX=0.0,
                 sigmaY=0.0,
                 sigmaZ=0.0,
                 sigmaXprime=1e-6,
                 sigmaZprime=1e-6,)
    beam0 = S4Beam()
    beam0.generate_source(source)
    print(beam0.info())

    if do_plot:
        plotxy(beam0, 4, 6, title="Image 0", nbins=201)


    boundary_shape = None

    mirror1 = S4ToroidMirrorElement(optical_element=S4ToroidMirror(name="M1",
                                                                     surface_calculation=SurfaceCalculation.EXTERNAL,
                                                                     min_radius=0.157068,
                                                                     maj_radius=358.124803 - 0.157068,
                                                                     f_torus=0,
                                                                     boundary_shape=boundary_shape),
                                      coordinates=ElementCoordinates(p = 10.0,
                                                                     q = 6.0,
                                                                     angle_radial = numpy.radians(88.8)),
                                      input_beam=beam0)


    print(mirror1.info())

    beam1, mirr1 = mirror1.trace_beam()
    print(mirr1.info())

    if do_plot:
        plotxy(beam1, 1, 3, title="Image 1", nbins=101, nolost=1)
        plotxy(mirr1, 2, 1, title="Footprint 1", nbins=101, nolost=1)


if __name__ == "__main__":


    do_plot = True

    example_branch_2(do_plot=do_plot) # toroid

