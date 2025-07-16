import numpy
from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.beam.s4_beam import S4Beam
from shadow4.sources.source_geometrical.source_gaussian import SourceGaussian
from syned.beamline.shape import  Ellipse
from shadow4.beamline.optical_elements.mirrors.s4_conic_mirror import S4ConicMirror, S4ConicMirrorElement
from shadow4.beamline.optical_elements.absorbers.s4_screen import S4Screen, S4ScreenElement

def example_branch_1():
    source = SourceGaussian(nrays=550000,
                 sigmaX=0.0,
                 sigmaY=0.0,
                 sigmaZ=0.0,
                 sigmaXprime=1e-6,
                 sigmaZprime=1e-6,)
    beam0 = S4Beam()
    beam0.generate_source(source)
    print(beam0.info())

    boundary_shape = Ellipse(a_axis_min=0, a_axis_max=1e-05, b_axis_min=-0.0005, b_axis_max=0)
    coordinates_syned = ElementCoordinates(p = 10.0,
                                           q = 6.0,
                                           angle_radial = numpy.radians(88.8))

    mirror1 = S4ConicMirrorElement(optical_element=S4ConicMirror(name="M1",
                                                       conic_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                                                       boundary_shape=boundary_shape),
                              coordinates=coordinates_syned,
                              input_beam=beam0)


    
    beam1, _ = mirror1.trace_beam()

    s1 = S4ScreenElement(optical_element=S4Screen(),
                         coordinates=ElementCoordinates(p=0.0, q=1.0, angle_radial=0.0),
                         input_beam=beam1)

    beam_on_source, _ = s1.trace_beam()

    rays = beam_on_source.rays_good
    x = rays[:, 0]
    z = rays[:, 2]
    
    image_data, _, _ = numpy.histogram2d(x, z, bins=201)
    plt.imshow(image_data, interpolation='nearest', origin='lower',)
    plt.show()

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    example_branch_1()

