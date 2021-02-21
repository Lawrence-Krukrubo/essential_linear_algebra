from .vector import Vector
from .decimal_ import MyDecimal
import numpy as np
import matplotlib.pyplot as plt


class Lines2D(object):
    """ Class for creating and manipulating 2D line objects,
        exploring their properties and attributes."""

    X, Y = None, None

    def __init__(self, coefficients, constant_term):
        self.coefficients = coefficients
        self.constant_term = constant_term
        self.dimension = 2
        self.normal_vector = Vector(self.coefficients)

        num = [int, float, np.int32, np.float32, np.int64, np.float64]
        assert self.dimension == len(self.coefficients), 'len(coefficients) Must Be == 2!'
        assert type(self.constant_term) in num, 'Constant-Term Must Be a Number!'
        for i in self.coefficients:
            assert type(i) in num, 'Coefficients  Must Be Numerical!'

    def __str__(self):
        ret = 'Lines2D Object:\n'
        temp = f'Equation:\n {self.coefficients[0]}x {self.coefficients[1]}y = {self.constant_term}'
        return ret+temp

    def __eq__(self, other):
        self_other_constants = self.constant_term == other.constant_term
        return self_other_constants and self.normal_vector.__eq__(other.normal_vector)

    def is_parallel_to(self, other):
        """Confirm if two lines are parallel

        Two lines are parallel if the coefficients
        of one line is a scalar multiple of the
        other line, irrespective of absolute values.

        Usage Example:
                    equation1 = 2x + 3y = 2
                    equation2 = 3x + 2y = 5

                    represent both equations as
                    Lines2D objects...
                    line1 = Lines2D((2, 3), 2)
                    line2 = Lines2D((3, 2), 5)

                    line1.is_parallel_to(line2)
                    >> False

        :param other: A line in 2D, same as self.
        :return: True or False
        """
        assert self.dimension == other.dimension, 'Dimensions Must Be Equal!'

        a, b = self.normal_vector, other.normal_vector
        if a.is_zero_vector(b):
            return True

        val = round((a.coordinates[0] / b.coordinates[0]), 2)
        for x, y in zip(a.coordinates, b.coordinates):
            try:
                assert round((x / y), 2) == val
            except AssertionError:
                return False

        return True

    def dir_vec(self):
        """Given a line, find it's
            direction vector

            The direction vector in 2D
            can be got by reversing the
            coefficients of a line and
            negating one.

            Usage Example:
                    equation1 = 2x + 3y = 2

                    represent the equation as
                    a Lines2D object...
                    line1 = Lines2D((2, 3), 2)

                    line1.dir_vec()
                    >> Vector([3, -2])

        :return: a Vector object, the direction vector
        """

        # First assert i`t's not the zero-vector
        try:
            assert not MyDecimal.is_near_zero(sum(self.coefficients))
        except AssertionError:
            return 0

        a = self.normal_vector
        x, y = self.coefficients[-1], -(self.coefficients[0])
        b = Vector([x, y])

        if a.is_orthogonal_to(b):
            return b

        return None

    def find_y_intercept(self):
        """Given a line in 2D
            find the y-intercept

            Find the value of y,
            when x is 0 (y-intercept).

        :return: a Tuple of x,y (Int or Float) coordinates
        """
        # Let's assume x = 0
        # to find y, we substitute
        x, y = 0, 0

        # use a try-catch block in case of zero-division error
        try:
            y = round(self.constant_term / self.coefficients[-1], 3)
        except ZeroDivisionError:
            pass
        return x, y

    def find_x_intercept(self):
        """Given a line in 2D
            find the x-intercept.

            Find the value of x,
            when y is 0 (x-intercept).

        :return: a Tuple of x,y coordinates
        """
        # Let's assume y = 0
        # to find , we substitute
        x, y = 0, 0

        # use a try-catch block in case of zero-division error
        try:
            x = round(self.constant_term / self.coefficients[0], 3)
        except ZeroDivisionError:
            pass
        return x, y

    def find_slope_and_intercept(self):
        """Find the slope and intercept of
            a line in 2D

            The slope is simply the rise over
            the run. (y1-y2 / x1-x2)

            Usage Example:
                    equation1 = 2x + 3y = 2

                    represent the equation as
                    a Lines2D object...
                    line1 = Lines2D((2, 3), 2)

                    line1.find_slope_and_intercept()
                    >> <slope, intercept>

        :return: Return a Tuple of slope and intercept
                these must be Ints or Floats.
        """
        point1 = self.find_y_intercept()
        point2 = self.find_x_intercept()

        slope, intercept = 0, round(point1[-1], 4)

        # use a try-catch block in case of zero-division error
        try:
            slope = round((point1[-1] - point2[-1]) / (point1[0] - point2[0]), 4)
        except ZeroDivisionError:
            pass

        return slope, intercept

    def is_equal_to(self, other):
        """Assert that two parallel lines
            are equal and the same

        Two parallel lines in 2D are equal/same if
        The direction vector of one line
        is orthogonal to the normal vectors
        of both lines.

        Usage Example:
                    equation1 = 2x + 3y = 2
                    equation2 = 3x + 2y = 5

                    represent both equations as
                    Lines2D objects...
                    line1 = Lines2D((2, 3), 2)
                    line2 = Lines2D((3, 2), 5)

                    line1.is_equal_to(line2)
                    >> False

        :param other: a line with same dim as self
        :return: True or False
        """
        try:
            assert self.is_parallel_to(other)
        except AssertionError:
            return False
        # find one point on self and other
        self_point = self.find_y_intercept()
        other_point = other.find_y_intercept()

        # find the vector between those points
        points_vec = (self_point[0] - other_point[0], self_point[1] - other_point[1])
        points_vec = Vector(points_vec)

        return points_vec.is_orthogonal_to(other.normal_vector)

    def radians(self, vec2):
        """This method calculates the angle between two lines
                in radians and returns a float.

        The angle is the arc-cosine of the dot-product of the two,
        vectors, divided by the product of their magnitudes.
        """

        return self.normal_vector.radians(vec2.normal_vector)

    def degrees(self, vec2):
        """This method calculates the angle between two lines
                in degrees and returns a float.

        Simply call the radians method on the vectors
        and multiply the radians value by (180/pi) to get degrees.
        """

        return self.normal_vector.degrees(vec2.normal_vector)

    def plot(self):
        """Plot a line in 2D,

        This Method is called on a Lines2D object.

        Usage Example:
                    equation1 = 2x + 3y = 2

                    represent equation1 as
                    Lines2D objects...
                    line1 = Lines2D((2, 3), 2)

                    line1.plot()

        :return: None (just plots the line in 2D)
        """
        slope, intercept = self.find_slope_and_intercept()
        slopes, intercepts = [], []
        slopes.append(slope)
        intercepts.append(intercept)

        X, Y, const = self.coefficients[0], self.coefficients[1], self.constant_term
        x = np.linspace(-5, 5, 500)

        for slope, intercept in zip(list(slopes), list(intercepts)):
            plt.plot(x, x * slope + intercept, '-r')
        plt.title(f'Linear Equation: {X}x {+Y}y = {const}', fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12, rotation=1.4)

        plt.grid(linestyle='dotted')
        plt.show()


if __name__ == '__main__':
    one = Lines2D((4.046, 2.836), 1.21)
    two = Lines2D((10.115, 7.09), 3.025)

    three = Lines2D((7.204, 3.182), 8.68)
    four = Lines2D((8.172, 4.114), 9.883)

    five = Lines2D((1.182, 5.562), 6.744)
    six = Lines2D((1.773, 8.343), 9.525)

