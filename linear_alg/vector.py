import math
import numpy as np
from .points import Point


class Vector(object):
    """ Class for creating and manipulating vector objects,
        exploring vector properties and attributes."""

    def __init__(self, coordinates):
        """ Initialise the Vector Class

        @Param:
        coordinates are vector coordinates that:-
        Must be a list or tuple, containing int or float elements.

        """
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = [round(i, 4) for i in coordinates]
            self.dimension = len(coordinates)
        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

        num = [int, float, np.int32, np.float32, np.int64, np.float64]
        for i in self.coordinates:
            assert type(i) in num, 'Coefficients  Must Be Numerical!'

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, other):
        return self.coordinates == other.coordinates

    def add(self, *args):
        """ Add arbitrary number of vectors

        Vector addition is commutative and element-wise
        addition of corresponding coordinates

        For Example:
        vec1 = Vector([1,2])
        vec2 = Vector([3,4])
        vec3 = Vector([5,6])

        vec1.add(vec2)
        >> Vector([4,6])

        vec1.add([vec2, vec3])
        >> Vector([9,12])

        @:param: args must be a vector object or a list of
                vectors, each of equal dimension.
        @:return: a vector from sum of all coordinates
        """
        try:
            for i in args:
                assert self.dimension == i.dimension
                count = 0
                for j in i.coordinates:
                    self.coordinates[count] += j
                    count += 1
        except AssertionError as e:
            return e, 'vectors must have same dimensions'

        return self

    def minus(self, *args):
        """ Minus arbitrary number of vectors

        Vector subtraction is non-commutative and
        element-wise subtraction of corresponding coordinates

        For Example:
        vec1 = Vector([1,2])
        vec2 = Vector([3,4])
        vec3 = Vector([5,6])

        vec1.minus(vec2)
        >> Vector([-2,-2])

        vec1.minus([vec2, vec3])
        >> Vector([-7,-8])

        @:param: args must be vectors of equal dimensions
        @:return: a vector from subtraction of all coordinates
        """
        try:
            for i in args:
                assert self.dimension == i.dimension
                count = 0
                for j in i.coordinates:
                    self.coordinates[count] -= j
                    count += 1
        except AssertionError as e:
            return e, 'vectors must have same dimensions'

        return self

    def scalar_multiply(self, scalar):
        """ Scalar multiply a vector

        Note that multiplying a vector by a negative number,
        causes the vector to point in the opposite direction,
        as well as possibly changing its' magnitude.

        For Example:
                    vec1 = Vector([1,2, 3])
                    scalar = 4

                    vec1.scalar_multiply(4)
                    >>Vector([4, 8, 12])

        :param: scalar must be an int or float, pos or neg.
        :return: Vector with coordinates multiplied by scalar,
                each coordinate rounded off to 4 D.P.
        """

        for i in range(len(self.coordinates)):
            self.coordinates[i] *= scalar
            self.coordinates[i] = round(self.coordinates[i], 4)

        return self

    def magnitude(self):
        """Compute the magnitude or length of a vector.

        The magnitude of a vector is the square root of,
        calling the dot-product on itself.

        Usage Example:
                    vec1 = Vector([2,3])
                    vec1.magnitude()
                    >> <Some-Numeric-Value>

        :return: Returns a scalar of type int or float
        """
        dot_multiply = sum([i**2 for i in self.coordinates])
        magnitude = math.sqrt(dot_multiply)

        return round(magnitude, 4)

    def unit_vector(self):
        """Compute the unit vector.

        To compute the unit vector,
        First, compute the vector magnitude,then
        multiply the inverse of the magnitude
        by the vector.
        This method does all that for you.

        Usage Example:
                    vec1 = Vector([2,3])
                    vec1.unit_vector()
                    >> <Some-Vector>

        :return: Return the unit vector
        """
        magnitude = self.magnitude()

        # If magnitude > 0, meaning not the zero vector,
        # then return unit vector, else return 0
        if magnitude:
            inv = 1 / magnitude
            return self.scalar_multiply(inv)
        else:
            return 0

    def is_zero_vector(self, vec2=None, tolerance=1e-7):
        """Check if a vector is a zero-vector.
            If called on a vector and given another
            vector as parameter, the method checks if
            either of the 2 vectors is the zero vector.

        The zero-vector is a vector that has a
        magnitude of zero. It is both parallel and
        orthogonal to itself and all other vectors.

        For example, if 2 vectors exist:
            vec1 = Vector([2,3])
            vec2 = Vector([0,0])

            vec1.is_zero_vector()
            >> False

            vec1.is_zero_vector(vec2)
            >>True

            vec2.is_zero_vector()
            >>True

        :param vec2: a vector with same dimension as self
        :param tolerance: A minute floating limit to accommodate
                        small floating point differences for zero.
                        default tolerance=1e-7.

                        if you want to set a custom tolerance,
                        simply pass it in the function call like:-
                        tolerance = <Your Specific Tolerance>

        :return: True or False
        """
        if vec2:
            try:
                assert self._has_equal_dim(vec2)
            except AssertionError as e:
                return e
            return abs(self.magnitude()) < tolerance or abs(vec2.magnitude()) < tolerance

        return abs(self.magnitude()) < tolerance

    def _has_equal_dim(self, vec2):
        """Asserts two vectors have equal dimensions

        :param vec2: A vector that should have same dim as self
        :return: returns True or False
        """

        return self.dimension == vec2.dimension

    def dot_product(self, vec2):
        """This method returns the dot-product of two vectors.

        The dot-product is the sum of element-wise multiplication
        of both vectors. It's commutative and returns a number.

        Usage Example:
                    vec1 = Vector([2,3])
                    vec2 = Vector([3,4])

                    vec1.dot_product(vec2)
                    >> (2*3) + (3*4) = 18

        @:param: vec2 is a vector of equal dimension with self
        @:return: A float or int
        """
        try:
            if self.is_zero_vector(vec2):
                return 0
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        def dot_multiply(vec_x, vec_y):
            """A recursive function to multiply
            corresponding elements of two vectors

            :param vec_x: the first vector
            :param vec_y: the second vector
            :return: A new vector with each element a
                     product of corresponding elements
            """
            new_list = []
            if not vec_x:
                return new_list
            new_list.append(vec_x[0] * vec_y[0])
            return new_list + dot_multiply(vec_x[1:], vec_y[1:])

        dot_product = round(sum(dot_multiply(self.coordinates, vec2.coordinates)), 4)

        return dot_product

    def radians(self, vec2):
        """This method calculates the angle between two vectors
                in radians.

        The angle is the arc-cosine of the dot-product of the two,
        vectors, divided by the product of their magnitudes.

        Usage Example:
                    vec1 = Vector([2,3])
                    vec2 = Vector([3,4])

                    vec1.radians(vec2)
                    >> <Some-Numeric-Value>

        @:param vec2: A vector object with same dimension as self
        @:return: an Int or Float (the angle in radians)
        """
        try:
            if self.is_zero_vector(vec2):
                return 0
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        dot_product = round(self.dot_product(vec2))
        magnitudes_multiply = round(self.magnitude() * vec2.magnitude())
        theta = round(math.acos(dot_product / magnitudes_multiply), 4)

        return theta

    def degrees(self, vec2):
        """This method calculates the angle between
            two vectors in degrees.

        It simply calls the radians method on the vectors
        and multiplies the radians value by (180/pi) to get degrees.

        Usage Example:
                    vec1 = Vector([1,2])
                    vec2 = Vector([3,4])

                    vec1.degrees(vec2)
                    >> <Some-Numeric-Value>

        @:param vec2: A vector object with same dimension as self
        @:return: an Int or Float (the angle in degrees)
        """
        radian = self.radians(vec2)
        degree = round(radian * (180 / math.pi), 4)

        return degree

    def is_parallel_to(self, vec2):
        """Check if one vector is a scalar multiple,
        of the other vector and vice versa

        Two vectors are parallel if one is a scalar
        multiple of the other. If the scalar is
        a negative number, then both vectors will be
        opposite and have angle of 180 degrees between.
        Else, both vectors will have angle of 0 degrees
        as they point in the same direction.

        Usage Example:
                    vec1 = Vector([1,2])
                    vec2 = Vector([2,4])
                    vec3 = Vector([0,5])

                    vec1.is_parallel_to(vec2)
                    >> True

                    vec1.is_parallel_to(vec3)
                    >> False

        :param vec2: A vector of same dimension as self
        :return: Return a boolean True or False
        """
        try:
            if self.is_zero_vector(vec2):
                return True
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        is_180 = round(self.degrees(vec2)) == 180
        is_zero = round(self.degrees(vec2)) == 0

        return (is_180 + is_zero) == 1

    def is_orthogonal_to(self, vec2):
        """Check if two vectors are orthogonal

        Two vectors are orthogonal if their dot-product is 0.
        This usually happens if one is a zero-vector or they are at
        right angles to each other

        Usage Example:
                    vec1 = Vector([1,2])
                    vec2 = Vector([2,4])
                    vec3 = Vector([0,0])

                    vec1.is_orthogonal_to(vec2)
                    >> False

                    vec1.is_orthogonal_to(vec3)
                    >> True

        :param vec2: Vector with same dimension as self
        :return: True or False
        """
        try:
            if self.is_zero_vector(vec2):
                return True
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        return round(self.degrees(vec2)) == 90

    def v_parallel(self, vec2):
        """Find the component of vector self,
         parallel to the basis vector(vec2),
         given that self is projected on vec2.

        To compute v_parallel, we multiply the unit_vector
        of vec2(basis vector), by the dot-product
        of the unit_vector of vec2 and self

        Usage Example:
                    vec1 = Vector([1,2])
                    vec2 = Vector([2,4])

                    vec1.v_parallel(vec2)
                    >> <Some-Vector>

        :param vec2: A vector with same dimension as self
        :return: A vector (v_parallel)
        """
        try:
            if self.is_zero_vector(vec2):
                return 0
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        unit_vec2 = vec2.unit_vector()
        self_dot_unit_vec2 = self.dot_product(unit_vec2)
        v_para = unit_vec2.scalar_multiply(self_dot_unit_vec2)

        return v_para

    def v_perp(self, vec2):
        """ Find the component of vector self orthogonal
        to vec2, given that self is projected on vec2

        Any non-zero vector can be represented as the
        sum of its component orthogonal/perpendicular
        to the basis vector(vec2) and its component
        parallel to the basis vector

        Usage Example:
                    vec1 = Vector([1,2])
                    vec2 = Vector([2,4])

                    vec1.v_perp(vec2)
                    >> <Some-Vector>

        :param vec2: Vector with same dimension as self
        :return: Vector (v_perp)
        """
        try:
            if self.is_zero_vector(vec2):
                return 0
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        v_parallel = self.v_parallel(vec2)
        v_perp = self.minus(v_parallel)

        return v_perp

    def cross_product(self, vec2):
        """Find the cross-product of self and vec2

        The cross product is non-commutative.
        Both vectors must be <= 3 dimensions.
        cross_product is useful for computing
        the area of the parallelogram spanned by
        these 2 vectors.
        Cross product returns a vector,
        which must be orthogonal to both vectors.

        Ideally, both vectors should be 3D, but
        If either vector dimension is less than
        3D, zero-padding is appended.

        Usage Example:
                    vec1 = Vector([1,2,3])
                    vec2 = Vector([2,4,5])

                    vec1.cross_product(vec2)
                    >> <Some-Vector>

        :param vec2: vector with 3 dimension equal to self
                    if dim < 3, append dim of 0.
        :return: Vector (cross-product)
        """
        try:
            assert self.dimension <= vec2.dimension <= 3
            while True:
                if self.dimension + vec2.dimension == 6:
                    break
                if self.dimension < 3:
                    self.coordinates.append(0)
                    self.dimension += 1
                if vec2.dimension < 3:
                    vec2.coordinates.append(0)
                    vec2.dimension += 1
        except AssertionError:
            return 'DimensionError: dim must be <= 3!'
        x1, y1, z1 = self.coordinates
        x2, y2, z2 = vec2.coordinates

        cross_vec = Vector([0, 0, 0])
        cross_vec.coordinates[0] = (y1 * z2) - (y2 * z1)
        cross_vec.coordinates[1] = -((x1 * z2) - (x2 * z1))
        cross_vec.coordinates[2] = (x1 * y2) - (x2 * y1)
        try:
            assert self.is_orthogonal_to(cross_vec)
            assert vec2.is_orthogonal_to(cross_vec)
        except AssertionError:
            return 'Cross-Vector must be orthogonal to both vectors!'

        return cross_vec

    def area_of_parallelogram(self, vec2):
        """Return the area of the parallelogram
        spanned by two vectors

        The area of the parallelogram spanned by
        two vectors is simply the magnitude of
        the cross-product of these two vectors.
        Both vectors must be <= 3D.

        Ideally, both vectors should be 3D, but
        If either vector dimension is less than
        3D, zero-padding is appended.

        Usage Example:
                    vec1 = Vector([1,2,3])
                    vec2 = Vector([2,4,5])

                    vec1.area_of_parallelogram(vec2)
                    >> <Some-Numerical-Value>

        Zero-appended Example:
                    vec1 = Vector([1,2])
                    vec2 = Vector([2,4,5])

                    On the function call, vec1 becomes...
                    vec1 = Vector([1,2,0])

                    vec1.area_of_parallelogram(vec2)
                    >> <Some-Numerical-Value>

        :param vec2: vector of no more than 3 dimension
                    same as self.
        :return: A number (magnitude of cross-product)
        """
        cross_vector = self.cross_product(vec2)
        parallelogram_area = cross_vector.magnitude()

        return round(parallelogram_area, 4)

    def area_of_triangle(self, vec2):
        """Return the area of the triangle
        spanned by two vectors

        The area of the triangle spanned by
        two vectors is simply the magnitude of
        the cross product of these two vectors,
        divided by 2. Both vectors must be <= 3D.

        Ideally, both vectors should be 3D, but
        If either vector dimension is less than
        3D, zero-padding is appended.

        Usage Example:
                    vec1 = Vector([1,2,3])
                    vec2 = Vector([2,4,5])

                    vec1.area_of_triangle(vec2)
                    >> <Some-Numerical-Value>

        Zero-appended Example:
                    vec1 = Vector([1,2])
                    vec2 = Vector([2,4,5])

                    On the function call, vec1 becomes...
                    vec1 = Vector([1,2,0])

                    vec1.area_of_triangle(vec2)
                    >> <Some-Numerical-Value>

        :param vec2: vector of no more than 3 dimension
                    same as self.
        :return: A number (area of triangle of 2 vectors)
        """
        cross_vector = self.cross_product(vec2)
        area_of_parallelogram = cross_vector.magnitude()
        triangle_area = area_of_parallelogram / 2.

        return round(triangle_area, 4)

    @staticmethod
    def get_dir_vec(point1, point2):
        """Given two Points, get the Vector
            that connects point1 to point2.
            Both points must have same dimension.

            The Direction Vector is any Vector that
            connects 2 points on a line, plane or hyper-plane.

            For Example in 2D:
                    point1 = (1, 2)
                    point2 = (3, 4)
                    Point.get_dir_vec(point1, point2)
                    >> Vector([2, 2])

            For Example in 3D:
                    point1 = (1, 2, 3)
                    point2 = (3, 4, 5)
                    Point.get_dir_vec(point1, point2)
                    >> Vector([2, 2, 2])

        :param point1: A tuple or triple of Int or Float
        :param point2: A tuple or triple of Int or Float
        :return: a Vector object
        """
        check = 0
        try:
            assert len(point1) == len(point2) >= 2
            check += 1
            assert not point1 == point2
        except AssertionError:
            if check:
                return 'ERROR: Point1 and Point2 Must not be Equal'
            return 'ERROR: Points Must Have Same Dimension >= 2.'

        coordinates = []

        for i, j in zip(point2, point1):
            coordinates.append(i - j)

        return Vector(coordinates)

    @staticmethod
    def plot_dir_vec(point1, point2):
        """Given two Points in 2D or 3D
            plot a Vector from the 1st point
            to the 2nd point. Both points must
            have the same dimension: 2D or 3D.

            The Direction Vector is any Vector that
            connects 2 points on a line, plane or hyper-plane.

            For Example in 2D:
                    point1 = (1, 2)
                    point2 = (3, 4)
                    Point.plot_dir_vec(point1, point2)

            For Example in 3D:
                    point1 = (1, 2, 3)
                    point2 = (3, 4, 5)
                    Point.plot_dir_vec(point1, point2)

        :param point1: A tuple or triple of Int or Float
        :param point2: A tuple or triple of Int or Float
        :return: None (Plots the vector connecting both points)
        """
        check = 0
        try:
            assert 2 <= len(point1) == len(point2) <= 3
            check += 1
            assert not Point.__eq__(point1, point2)
        except AssertionError:
            if check:
                return 'ERROR: Point1 and Point2 Must not be Equal'
            return 'ERROR: Points Must Have Same Dimension (2D/3D).'

        if len(point1) == 2:
            Point.plot_points_2d_vec(point1, point2)
        else:
            Point.plot_points_3d_vec(point1, point2)

    def plot(self):
        """Plot a vector in 2D or 3D.

        :return: None (just plots the vector)
        """

        try:
            assert 2 <= self.dimension <= 3
        except AssertionError:
            return "ERROR: Vector Dimension Must be 2D or 3D"

        point1 = list(np.round(np.random.uniform(low=-5, high=5, size=(self.dimension,))))
        point2 = [i + j for i, j in zip(point1, self.coordinates)]

        Vector.plot_dir_vec(point1, point2)