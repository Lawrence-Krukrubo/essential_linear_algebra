import numpy as np
from .decimal_ import MyDecimal
from .lines_2d import Lines2D
from .planes_3d import Planes3D
from .hyper_planes import HyperPlanes


class GeHelper():
    """Class of helper functions
        for the Gaussian-Elimination
        Algorithm
        """
    def __init__(self, dimension, plane_objects):
        self.dimension = dimension
        self.plane_objects = plane_objects

        # assert all planes are in same dimension
        for object in self.plane_objects:
            assert self.dimension == len(object.coefficients), \
                "Dimensions must be Equal"

    def sort_plane_objects(self):
        """Sort the planes by count
            of zero-leading index

        :return: self
        """
        sort_list = []
        for ind, plane in enumerate(self.plane_objects):
            temp = list(plane.coefficients)
            while True:
                if not temp:
                    break
                if MyDecimal.is_near_zero(temp[0]):
                    del temp[0]
                else:
                    break
            sort_list.append(len(temp))

        y = []
        count = len(sort_list)
        while count > 0:
            max_value = max(sort_list)
            max_index = sort_list.index(max_value)
            y.append(max_index)
            sort_list[max_index] = -(float('inf'))
            count -= 1

        # finally re-arrange self.plane_objects according
        # To the linear-equations with fewer leading zeros
        for ind, position in enumerate(y):
            sort_list[ind] = self.plane_objects[position]

        self.plane_objects = sort_list

    def first_non_zero_index(self):
        """Find the first non-zero-index
            for each planes coefficients
        :return:
        """
        check = [-(float('inf'))] * len(self.plane_objects)
        for ind, plane in enumerate(self.plane_objects):
            for i, j in enumerate(plane.coefficients):
                if not MyDecimal.is_near_zero(j):
                    check[ind] = i
                    break

        return check

    def is_inconsistent(self):
        """Check if no intersection

        A system of equations involving 2 or
        more planes will have no solution using
        the Gaussian Elimination Algorithm, if at
        any point we have 0 as coefficients and
        non-zero on the constant-term

        :return: True or false
        """
        for plane in self.plane_objects:
            s = sum(plane.coefficients)
            k = plane.constant_term
            if MyDecimal.is_near_zero(s) and k:
                return True

        return False

    def _update_planes(self, temp, const):
        """A helper function to update the
            new values of the system of
            equations after some computation

        :param temp: The new coefficients
        :param const: the new constant term
        :return: A plane/hyper-plane with both
            temp and const values as coefficients
            and constant terms.
        """
        if self.dimension < 3:
            x = Lines2D(temp, const)
        elif self.dimension > 3:
            x = HyperPlanes(self.dimension, temp, const)
        else:
            x = Planes3D(temp, const)

        return x

    def multiply_row(self, row, scalar):
        """Multiply a system of equation by a scalar

        :param row: A system of equation
        :param scalar: A given coefficient int or float
        :return: None
        """
        x = list(self.plane_objects[row].coefficients)
        i = self.plane_objects[row].constant_term

        x = np.array(x+[i]) * scalar
        temp, const = tuple(x[:-1]), x[-1]
        val = self._update_planes(temp, const)

        self.plane_objects[row] = val

    def divide_row(self, row, scalar):
        """Divide a system of equation by a scalar

        :param row: A system of equation
        :param scalar: A given coefficient int or float
        :return: None
        """

        x = list(self.plane_objects[row].coefficients)
        i = self.plane_objects[row].constant_term
        x = np.array(x+[i]) / scalar

        temp, const = tuple(x[:-1]), x[-1]
        val = self._update_planes(temp, const)

        self.plane_objects[row] = val

    def subtract_rows(self, row_to_subtract, row_to_be_subtracted_from):
        """Subtract one plane/hyper-plane from another
        :param row_to_subtract: A plane or hyper-plane
        :param row_to_be_subtracted_from: A plane/hyper-plane
        :return:
        """
        x = list(self.plane_objects[row_to_subtract].coefficients)
        i = self.plane_objects[row_to_subtract].constant_term
        x = np.array(x+[i])

        y = list(self.plane_objects[row_to_be_subtracted_from].coefficients)
        i = self.plane_objects[row_to_be_subtracted_from].constant_term
        y = np.array(y+[i])

        sub = y - x
        temp, const = tuple(sub[:-1]), sub[-1]
        val = self._update_planes(temp, const)

        self.plane_objects[row_to_be_subtracted_from] = val

    def round_off(self, lim=6):
        """Multiply a system of equation by a scalar

        :param row: A system of equation
        :param scalar: A given coefficient int or float
        :return: None
        """
        for ind, plane in enumerate(self.plane_objects):
            x = list(plane.coefficients)
            i = plane.constant_term

            x = np.array(x+[i])
            x = np.round(x, lim)
            temp, const = tuple(x[:-1]), x[-1]
            val = self._update_planes(temp, const)

            self.plane_objects[ind] = val