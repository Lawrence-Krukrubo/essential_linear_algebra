from .vector import Vector
from .lines_2d import Lines2D
from .decimal_ import MyDecimal
import numpy as np


class HyperPlanes(object):
    """Class for creating and exploring HyperPlanes > 3D,
    finding Intersections, Coefficients, and other properties
    for HyperPlanes > 3-Dimensions.
    """
    NO_NONZERO_ELTS_FOUND_MSG = 'No Non-zero elements found!'

    def __init__(self, dimension, coefficients=None, constant_term=None):
        self.dimension = dimension

        if not coefficients:
            all_zeros = [0] * self.dimension
            coefficients = tuple(all_zeros)
        self.coefficients = coefficients
        if not constant_term:
            constant_term = 0
        self.constant_term = constant_term
        self.normal_vector = Vector(self.coefficients)

        assert self.dimension == len(self.coefficients) > 3, 'Dimension Must Be > 3'
        num = [int, float, np.int32, np.float32, np.int64, np.float64]
        assert type(self.constant_term) in num, 'Constant-Term Must Be a Number!'
        for i in self.coefficients:
            assert type(i) in num, 'Coefficients  Must Be Numerical!'

    def __eq__(self, other):
        """Assert if two hyperplanes have
        equal coefficients and constant terms

        :param other: A HyperPlane object
        :return: True or false
        """
        self_other_constants = self.constant_term == other.constant_term
        return self_other_constants and self.normal_vector.__eq__(other.normal_vector)

    def is_parallel_to(self, other):
        """Confirm if two hyper-planes are parallel

        Two hyper-planes are parallel if they have
        parallel normal vectors. Meaning their normal
        vectors are scalar multiples of each other
        irrespective of absolute value

        :param other: A plane with same dim as self
        :return:
        """
        self_2d = Lines2D(self.coefficients, self.constant_term)
        other_2d = Lines2D(other.coefficients, other.constant_term)

        return self_2d.is_parallel_to(other_2d)

    def find_point(self):
        """Given a hyper-plane in n-Dimension
            find any given point

        :return: a n-Tuple of coordinates
        """
        # Let's assume all coordinates except
        # the last coordinate is 0
        temp = [0] * self.dimension
        temp[-1] = self.constant_term / self.coefficients[-1]

        return tuple(temp)

    def is_equal_to(self, other):
        """Assert that two parallel hyper-planes
            are equal and the same

        Two parallel hyper-planes are equal/same,
        if the vector from any point in one hyper-plane
        to any point in the other hyper-plane is orthogonal
        to the normal vectors of either planes.

        :param other: a plane with same dim as self
        :return: True or False
        """
        try:
            assert self.is_parallel_to(other)
        except AssertionError:
            return False
        # find one point on self and other
        self_point = self.find_point()
        other_point = other.find_y_intercept()

        # find the vector between those points
        points_vec = [i - j for i, j in zip(self_point, other_point)]
        points_vec = Vector(points_vec)

        return points_vec.is_orthogonal_to(other.normal_vector)

    @staticmethod
    def first_nonzero_index(iterable):
        for k, item in enumerate(iterable):
            if not MyDecimal(item).is_near_zero():
                return k
        raise Exception(HyperPlanes.NO_NONZERO_ELTS_FOUND_MSG)