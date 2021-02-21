from .vector import Vector
from .decimal_ import MyDecimal
import numpy as np


class Planes3D(object):
    """Class for creating and exploring Planes in 3D,
    finding Intersections, Coefficients, and other properties
    for planes in 3-Dimensions.
    """
    X, Y, Z = None, None, None
    NO_NONZERO_ELTS_FOUND_MSG = 'No Non-zero elements found!'

    def __init__(self, coefficients=None, constant_term=None):
        self.dimension = 3

        if not coefficients:
            all_zeros = [0] * self.dimension
            coefficients = tuple(all_zeros)
        self.coefficients = coefficients
        if not constant_term:
            constant_term = 0
        self.constant_term = constant_term
        self.normal_vector = Vector(self.coefficients)

        assert self.dimension == len(self.coefficients), 'len(coefficients) Must Be == 3'
        num = [int, float, np.int32, np.float32, np.int64, np.float64]
        assert type(self.constant_term) in num, 'Constant-Term Must Be a Number!'
        for i in self.coefficients:
            assert type(i) in num, 'Coefficients  Must Be Numerical!'

    def __eq__(self, other):
        self_other_constants = self.constant_term == other.constant_term
        return self_other_constants and self.normal_vector.__eq__(other.normal_vector)

    def is_parallel_to(self, vec2):
        """Confirm if two planes are parallel

        Two planes are parallel if they have
        parallel normal vectors. This means
        their normal vectors are scalar multiple
        of each other and their normal vectors
        have an angle of 0 or 180 degrees between them

        Usage Example:
                    equation1 = 2x + 3y + 4z = 5
                    equation2 = 3x + 2y - 6z = 9

                    represent both equations as
                    Planes3D objects...
                    plane1 = Planes3D((2, 3, 4), 5)
                    plane2 = Planes3D((3, 2, -6), 9)

                    plane1.is_parallel_to(plane2)
                    >> False

        :param vec2: A plane with same dim as self
        :return:
        """
        parallel_degrees = [0, 180]

        return round(self.degrees(vec2)) in parallel_degrees

    def find_point(self):
        """Given a plane in 3D
            find any given point.
            Specifically, find the value of z,
            when x and y are zero.

            Usage Example:
                    equation1 = 2x + 3y + 4z = 8

                    represent equation1 as a
                    Planes3D object...
                    plane1 = Planes3D((2, 3, 4), 8)

                    plane1.find_point()
                    >> (0, 0, 2)

                    The method assumes x==y==0
                    Therefore z = 8/4 = 2

        :return: a Triple of x,y,z coordinates
        """
        # Let's assume x and y == 0
        # to find z, we substitute
        self.X, self.Y = 0, 0

        self.Z = self.constant_term / self.coefficients[-1]

        return self.X, self.Y, self.Z

    def is_equal_to(self, other):
        """Assert that two parallel planes
            are equal and the same

        Two parallel planes in 3D are equal/same,
        if the direction vector from any point in a plane
        to any point in the other plane is orthogonal
        to the normal vectors of either planes.

        Usage Example:
                    equation1 = 2x + 3y + 4z = 5
                    equation2 = 3x + 2y - 6z = 9

                    represent both equations as
                    Planes3D objects...
                    plane1 = Planes3D((2, 3, 4), 5)
                    plane2 = Planes3D((3, 2, -6), 9)

                    plane1.is_equal_to(plane2)
                    >> False

        :param other: a plane with same dim as self
        :return: True or False
        """
        try:
            assert self.is_parallel_to(other)
        except AssertionError:
            return False
        # find one point on self and other
        self_point = self.find_point()
        other_point = other.find_point()

        self_, other_ = self.coefficients, other.coefficients

        # find the vector between those points
        points_vec = [i - j for i, j in zip(self_point, other_point)]
        points_vec = Vector(points_vec)

        # Now find the cross-vector of these two planes
        # The cross-vector is orthogonal to both planes
        cross_vec = Vector(self_).cross_product(Vector(other_))

        # Return that cross_vec is the same as the vector
        # connecting these two points, which should be
        # The same as the zero vector given that both vectors,
        # self and other, are same and parallel.

        return points_vec.__eq__(cross_vec)

    def radians(self, vec2):
        """This method calculates the angle between two planes
                in radians and returns a float.

        The angle is the arc-cosine of the dot-product of the two,
        vectors, divided by the product of their magnitudes.
        """

        return self.normal_vector.radians(vec2.normal_vector)

    def degrees(self, vec2):
        """This method calculates the angle between two planes
                in degrees and returns a float.

        Simply call the radians method on the vectors
        and multiply the radians value by (180/pi) to get degrees.
        """

        return self.normal_vector.degrees(vec2.normal_vector)

    @staticmethod
    def first_nonzero_index(iterable):
        """This method takes an iterable
            and finds the index of the
            first non-zero element

        :param iterable: Any iterable object in Python
        :return: Int (index of first non-zero element)
        """
        for k, item in enumerate(iterable):
            if not MyDecimal(item).is_near_zero():
                return k
        raise Exception(Planes3D.NO_NONZERO_ELTS_FOUND_MSG)


if __name__ == '__main__':
    one = Planes3D((-0.412, 3.806, 0.728), -3.46)
    two = Planes3D((1.03, -9.515, -1.82), 8.65)

    three = Planes3D((2.611, 5.528, 0.283), 4.6)
    four = Planes3D((7.715, 8.306, 5.342), 3.76)

    five = Planes3D((-7.926, 8.625, -7.212), -7.952)
    six = Planes3D((-2.642, 2.875, -2.404), -2.443)

    print(one.is_parallel_to(two))
    print(one.is_equal_to(two))
    print(three.is_parallel_to(four))
    print(three.is_equal_to(four))
    print(five.is_parallel_to(six))
    print(five.is_equal_to(six))