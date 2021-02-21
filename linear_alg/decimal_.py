from decimal import Decimal


class MyDecimal(Decimal):
    """Class to instantiate a
        Decimal object.
    """
    def is_near_zero(self, eps=1e-5):
        """Confirm if an object has an
            absolute value, less than a given
            epsilon threshold, close to zero.

            Usage Example:
                        from .decimal_ import MyDecimal
                        var = 0.0000001
                        var2 = 0.00001

                        MyDecimal.is_near_zero(var)
                        >> True

                        MyDecimal.is_near_zero(var2)
                        >> False

        :param eps: A minute floating point limit used
                    to accommodate floating point leakages
                    on objects whose values may be zero.
        :return: True or False
        """
        return abs(self) < eps