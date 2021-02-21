import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Arrow3D(FancyArrowPatch):
    """Class to instantiate an arrow in 3D

    """
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._xyz = (x,y,z)
        self._dxdydz = (dx,dy,dz)

    def draw(self, renderer):
        x1,y1,z1 = self._xyz
        dx,dy,dz = self._dxdydz
        x2,y2,z2 = (x1+dx,y1+dy,z1+dz)

        xs, ys, zs = proj_transform((x1,x2),(y1,y2),(z1,z2), renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add a 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


class Point:
    """ Class for manipulating points in 2D and 3D,
        exploring distance and points properties.
    """

    @staticmethod
    def __eq__(*args):
        """Assert 2 or more points are equal

        2 or more points are equal iff they occupy
        the same location (having same coordinates).

        :param *args: two or more points in any dimension
        :return: Boolean, True or False
        """
        check = args[0]

        for i in args:
            try:
                assert i == check
            except AssertionError:
                return False

        return True

    @staticmethod
    def manhattan_distance(point1, point2):
        """Find the Manhattan distance between two points

        Manhattan distance or taxi-cab distance between
        two points is the sum of the absolute difference
        between corresponding coordinates of these two points.

        point1 and point2 must be of same dimension and
        each must be a tuple or a list.

        Example:
            import Point...

            # For 2D
            point1 = (1,2)
            point2 = (3,4)
            Point.manhattan_distance(point1, point2)
            >> <an-Int-or-Float>

            # For 3 and higher D
            point1 = (1, 2, 3)
            point2 = (4, 5,6)
            Point.manhattan_distance(point1, point2)
            >> <an-Int-or-Float>

        :param point1: A list or tuple of ints or floats
        :param point2: A list or tuple of ints or floats
        :return: An int or float of manhattan distance
        """
        # Assert both points have same dimension
        # And both points are of type tuple or list
        count = 1
        try:
            assert type(point1) and type(point2) in [tuple, list]
            count += 1
            assert len(point1) == len(point2)
        except AssertionError:
            if count == 1:
                return 'ERROR: point1 and point2 must be a tuple or list'
            else:
                return 'ERROR: point1 and point2 Must have Same Dimension'

        distance_ = 0

        for coord1, coord2 in zip(point1, point2):
            distance_ += abs(coord1 - coord2)

        return round(distance_, 4)

    @staticmethod
    def euclidean_distance(point1, point2):
        """Find the Euclidean distance between two points

            The Euclidean distance between 2 points is simply
            the magnitude of the vector connecting these points.

            point1 and point2 must be in same dimension and
            each must be a tuple or a list.

            Example:
                import Point...

                # For 2D
                point1 = (1,2)
                point2 = (3,4)
                Point.euclidean_distance(point1, point2)
                >> <an-Int-or-Float>

                # For 3 and higher D
                point1 = (1, 2, 3)
                point2 = (4, 5,6)
                Point.euclidean_distance(point1, point2)
                >> <an-Int-or-Float>

            :param point1: A list or tuple of ints or floats
            :param point2: A list or tuple of ints or floats
            :return: An int or float of euclidean distance
        """
        # Assert both points have same dimension
        # And both points are of type tuple or list
        count = 1
        try:
            assert type(point1) and type(point2) in [tuple, list]
            count += 1
            assert len(point1) == len(point2)
        except AssertionError:
            if count == 1:
                return 'ERROR: point1 and point2 must be a Tuple or List'
            else:
                return 'ERROR: point1 and point2 Must have Same Dimension'

        distance_ = 0

        for coord1, coord2 in zip(point1, point2):
            dist_squ = pow(coord1 - coord2, 2)
            distance_ += dist_squ

        distance_ = pow(distance_, 0.5)

        return round(distance_, 4)

    @staticmethod
    def __plot_point(point):
        """Plot one point in 2D or 3D

        Example:
                point1 = (2, 3)  # 2D
                Point.plot_point(point1)

                point2 = (2, 3, 4)  # 3D
                Point.plot_point(point2)

        :param point: A tuple or triple of Ints or Floats
        :return: None (Just plots the point)
        """
        x, y, z = None, None, None
        assert 2 <= len(point) <= 3, 'Points Must Have 2 or 3 Coordinates'

        if len(point) < 3:
            x, y = point
        else:
            x, y, z = point

        title_dict = {'size': 14, 'weight': 'bold'}
        label_dict = {'size': 12, 'weight': 'bold'}
        plt.style.use('seaborn-white')

        if z is None:
            plt.scatter(x, y)
            plt.xlim(min(x, y) - 1, max(x, y) + 1)
            plt.ylim(min(x, y) - 1, max(x, y) + 1)
            plt.title(f'Point in 2D: (X= {x}, Y= {y})', fontdict=title_dict)
            plt.xlabel('X', fontdict=label_dict)
            plt.ylabel('Y', fontdict=label_dict, rotation=1.4)
            plt.grid(linestyle='dotted')

        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x, y, z, c='r', marker='o')
            ax.set_xlabel('X', fontdict=label_dict)
            ax.set_ylabel('Y', fontdict=label_dict)
            ax.set_zlabel('Z', fontdict=label_dict)
            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)
            ax.set_zlim(auto=True)
            ax.set_title(f'Point in 3D: (X={x}, Y={y}, Z={z})', fontdict=title_dict)

        plt.show()

    @staticmethod
    def __plot_points(*args):
        """Plot multiple points in 2D or 3D

        All Points Must Either be 2D or 3D

                Example 2D:
                        point1 = (2, 3)
                        point2 = (3, 4)
                        point3 = (5, 6)
                        Point.plot_points(point1, point2, point3)

                Example 3D:
                        point1 = (2, 3, 4)
                        point2 = (3, 4, 5)
                        point3 = (5, 6, 7)
                        Point.plot_points(point1, point2, point3)

                :param args: Multiple tuples or triples of points
                :return: None (Just plots the point)
                """

        title_dict = {'size': 14, 'weight': 'bold'}
        label_dict = {'size': 12, 'weight': 'bold'}
        plt.style.use('seaborn-white')

        x_list = []
        y_list = []
        z_list = []

        check = len(args[0])
        for i in args:
            if check == 2:
                x_list.append(i[0])
                y_list.append(i[1])
            else:
                x_list.append(i[0])
                y_list.append(i[1])
                z_list.append(i[2])

        if not z_list:
            plt.scatter(x_list, y_list)
            plt.xlim(min(x_list)-1, max(x_list)+1)
            plt.ylim(min(y_list)-1, max(y_list)+1)
            plt.title(f'Points in 2D: (X= {x_list}, Y= {y_list})', fontdict=title_dict)
            plt.xlabel('X', fontdict=label_dict)
            plt.ylabel('Y', fontdict=label_dict, rotation=1.4)
            plt.grid(linestyle='dotted')

        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x_list, y_list, z_list, c='r', marker='o')
            ax.set_xlabel('X', fontdict=label_dict)
            ax.set_ylabel('Y', fontdict=label_dict)
            ax.set_zlabel('Z', fontdict=label_dict)
            ax.set_xlim(min(x_list)-1, max(x_list)+1)
            ax.set_ylim(min(y_list)-1, max(y_list)+1)
            ax.set_zlim(min(z_list)-1, max(z_list)+1)
            ax.set_title(f'Point in 3D: (X={x_list}, Y={y_list}, Z={z_list})', fontdict=title_dict)

        plt.show()

    @staticmethod
    def plot_points_2d_vec(point1, point2):
        """Given two Points in 2D
        plot a Vector from the 1st point
        to the 2nd point

            For Example in 2D:
                    point1 = (1, 2)
                    point2 = (3, 4)
                    Point.plot_points_2d_vec(point1, point2)

        :param point1: A tuple of Int or Float
        :param point2: A tuple of Int or Float
        :return: None (Plots the vector connecting the points)
        """
        point1 = [round(i, 1) for i in point1]
        point2 = [round(i, 1) for i in point2]
        head_length = 0.5
        x = [round(j - k, 1) for j, k in zip(point2, point1)]

        x_mag = (sum([i**2 for i in x]))**0.5
        dx, dy = [i / x_mag for i in x]
        x_mag = x_mag - head_length

        ax = plt.axes()
        title_dict = {'size': 14.5, 'weight': 'bold'}
        label_dict = {'size': 12.5, 'weight': 'bold'}
        plt.style.use('seaborn-white')

        if len(point1) == 2:
            ax.arrow(point1[0], point1[1], dx * x_mag, dy * x_mag,
                     head_width=0.4, head_length=head_length, fc='red', ec='black', linewidth=2)

            plt.grid()
            if x[1] < 0 <= x[0]:
                plt.xlim(point1[0] - 1, point1[0] + x[0] + 1)
                plt.ylim((point1[1] + x[1]) - 1, point1[1] + 1)
                plt.annotate(f'({point1[0]}, {point1[1]})', (point1[0], point1[1] + 0.15), fontweight='bold')
                plt.annotate(f'({point2[0]}, {point2[1]})', (point2[0], point2[1] - 0.25), fontweight='bold')
            elif x[1] >= 0 <= x[0]:
                plt.xlim(point1[0] - 1, point1[0] + x[0] + 1)
                plt.ylim(point1[1] - 1, point1[1] + x[1] + 1)
                plt.annotate(f'({point1[0]}, {point1[1]})', (point1[0], point1[1] - 0.15), fontweight='bold')
                plt.annotate(f'({point2[0]}, {point2[1]})', (point2[0], point2[1] + 0.15), fontweight='bold')
            elif x[0] < 0 <= x[1]:
                plt.xlim((point1[0] + x[0]) - 1, point1[0] + 1)
                plt.ylim(point1[1] - 1, (point1[1] + x[1]) + 1)
                plt.annotate(f'({point1[0]}, {point1[1]})', (point1[0], point1[1] + 0.15), fontweight='bold')
                plt.annotate(f'({point2[0]}, {point2[1]})', (point2[0], point2[1] + 0.15), fontweight='bold')
            else:
                plt.xlim((point1[0] + x[0]) - 1, point1[0] + 1)
                plt.ylim((point1[1] + x[1]) - 1, point1[1] + 1)
                plt.annotate(f'({point1[0]}, {point1[1]})', (point1[0], point1[1] - 0.15), fontweight='bold')
                plt.annotate(f'({point2[0]}, {point2[1]})', (point2[0], point2[1] - 0.25), fontweight='bold')

            plt.scatter([point1[0], point2[0]], [point1[1], point2[1]], color='black')
            plt.title(f'Vector:({x[0]}x, {x[1]}y)', fontdict=title_dict)
            plt.xlabel('X', fontdict=label_dict)
            plt.ylabel('Y', fontdict=label_dict, rotation=1.4)

        plt.show()

    @staticmethod
    def plot_points_3d_vec(point1, point2):
        """Given two Points in 3D
        plot a Vector from the 1st point
        to the 2nd point

            For Example in 3D:
                    point1 = (1, 2, 3)
                    point2 = (3, 4, 5)
                    Point.plot_points_3d_vec(point1, point2)

        :param point1: A triple of Int or Float
        :param point2: A triple of Int or Float
        :return: None (Plots the vector connecting the points)
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        point1 = [round(i, 1) for i in point1]
        point2 = [round(i, 1) for i in point2]
        coords = [round(i - j, 1) for i, j in zip(point2, point1)]
        dx, dy, dz = coords
        x, y, z = point1
        lim = []
        
        for point, direction in zip(point1, coords):
            if direction < 0:
                lim.append(point)
                lim.append(point+direction-1)
            else:
                lim.append(point)
                lim.append(point+direction+1)

        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_zlim(lim[4], lim[5])

        ax.arrow3D(x, y, z,
                   dx, dy, dz,
                   mutation_scale=20,
                   ec='green',
                   fc='red')
        ax.set_title(f'Vector:({dx}x, {dy}y, {dz}z)', fontsize=16, fontweight='bold')
        ax.set_xlabel('X', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y', fontsize=13, fontweight='bold')
        ax.set_zlabel('Z', fontsize=13, fontweight='bold')
        ax.text(x=x, y=y-0.25, z=z, s=f'({x},{y},{z})', fontsize=12, fontweight='bold')
        ax.text(x=x+dx,
                y=y+dy+0.25,
                z=z+dz,
                s=f'({round(x+dx,1)},{round(y+dy,1)},{round(z+dz,1)})',
                fontsize=12,
                fontweight='bold')
        fig.tight_layout()

        plt.show()

    @staticmethod
    def plot_points(*args):
        """Plot one or more points in 2D or 3D

            All Points Must Either be 2D or 3D

                Example 2D:
                        point1 = (2, 3)
                        point2 = (3, 4)
                        point3 = (5, 6)
                        Point.plot_points(point1, point2, point3)

                Example 3D:
                        point1 = (2, 3, 4)
                        point2 = (3, 4, 5)
                        point3 = (5, 6, 7)
                        Point.plot_points(point1, point2, point3)

        :param args: Multiple tuples or triples of points
        :return: None (Just plots the point)
        """
        check = len(args[0])
        for point in args:
            try:
                assert len(point) == check
                assert 2 <= check <= 3
            except AssertionError:
                return 'ERROR: All Dimensions Must == 2D or 3D'

        if len(args) == 1:
            Point.__plot_point(args[0])
        else:
            Point.__plot_points(*args)