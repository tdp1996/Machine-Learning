import math
from typing import Optional


class Array:
    def __init__(self, data):
        """
        Initializes the Array object with given data.

        Args:
            data (list): A list or list of lists containing the array elements.
        """
        if not isinstance(data, list):
            raise ValueError("Data must be a list or list of lists.")

        if all(isinstance(elem, list) for elem in data):
            if not all(len(row) == len(data[0]) for row in data):
                raise ValueError("All rows must have the same number of columns.")
            self.shape = (len(data), len(data[0]))
        elif all(isinstance(elem, (int, float)) for elem in data):
            self.shape = (len(data),)
        else:
            raise ValueError(
                "Data must be a list of lists (2D array) or a list of numbers (1D array)."
            )

        self.data = data

    def __repr__(self) -> str:
        """
        Returns the string representation of the Array object.

        Returns:
            str: The string representation of the array.
        """
        return f"Array({self.data})"

    def getitem(self, index):
        """
        Gets the element at the specified index.

        Args:
            index (int or tuple): The index of the element to get.

        Returns:
            int, float, or list: The element at the specified index.
        """
        return self.data[index]

    @classmethod
    def from_list(cls, data_list):
        """
        Creates an Array object from a list.

        Args:
            data_list (list): A list or list of lists containing the array elements.

        Returns:
            Array: An Array object.
        """
        return cls(data_list)

    def tolist(self):
        """
        Converts the Array object to a list.

        Returns:
            list: The list representation of the array.
        """
        return self.data

    def transpose(self):
        """
        Transposes the array.

        Returns:
            Array: A new Array object with the transposed data.
        """
        result = list(map(list, zip(*self.data)))
        return Array(result)

    def _elementwise_op(self, other, op):
        """
        Helper method to perform element-wise operations.

        Args:
            other (Array or int or float): Another Array object or a scalar.
            op (callable): A function to perform the operation.

        Returns:
            Array: A new Array object with the result of the operation.

        Raises:
            ValueError: If the shapes of the arrays are not compatible for the operation.
        """
        if isinstance(other, Array):
            if self.shape == other.shape:
                if len(self.shape) == 1:
                    return Array(
                        [op(self.data[i], other.data[i]) for i in range(len(self.data))]
                    )
                elif len(self.shape) == 2:
                    return Array(
                        [
                            [
                                op(self.data[i][j], other.data[i][j])
                                for j in range(self.shape[1])
                            ]
                            for i in range(self.shape[0])
                        ]
                    )
            elif (
                len(self.shape) == 2
                and len(other.shape) == 1
                and self.shape[1] == other.shape[0]
            ):
                return Array(
                    [
                        [
                            op(self.data[i][j], other.data[j])
                            for j in range(self.shape[1])
                        ]
                        for i in range(self.shape[0])
                    ]
                )
            elif (
                len(self.shape) == 1
                and len(other.shape) == 2
                and self.shape[0] == other.shape[1]
            ):
                return Array(
                    [
                        [
                            op(self.data[j], other.data[i][j])
                            for j in range(other.shape[1])
                        ]
                        for i in range(other.shape[0])
                    ]
                )
            else:
                raise ValueError(
                    f"operands could not be broadcast together with shapes {self.shape} {other.shape}"
                )
        else:
            if len(self.shape) == 1:
                return Array([op(self.data[i], other) for i in range(len(self.data))])
            else:
                return Array(
                    [
                        [op(self.data[i][j], other) for j in range(self.shape[1])]
                        for i in range(self.shape[0])
                    ]
                )

    def __add__(self, other):
        """
        Adds two arrays element-wise or adds a scalar to each element of the array.

        Args:
            other (Array or int or float): Another Array object or a scalar.

        Returns:
            Array: A new Array object with the result of the addition.

        Raises:
            ValueError: If the shapes of the arrays are not compatible for addition.
        """

        return self._elementwise_op(other, lambda x, y: x + y)

    def __sub__(self, other):
        """
        Subtracts one array from another element-wise or subtracts a scalar from each element of the array.

        Args:
            other (Array or int or float): Another Array object or a scalar.

        Returns:
            Array: A new Array object with the result of the subtraction.

        Raises:
            ValueError: If the shapes of the arrays are not compatible for subtraction.
        """
        return self._elementwise_op(other, lambda x, y: x - y)

    def __mul__(self, other):
        """
        Multiplies two arrays element-wise or multiplies each element of the array by a scalar.

        Args:
            other (Array or int or float): Another Array object or a scalar.

        Returns:
            Array: A new Array object with the result of the multiplication.
        """
        return self._elementwise_op(other, lambda x, y: x * y)

    def __truediv__(self, other):
        """
        Divides one array by another element-wise or divides each element of the array by a scalar.

        Args:
            other (Array or int or float): Another Array object or a scalar.

        Returns:
            Array: A new Array object with the result of the division.
        """
        return self._elementwise_op(other, lambda x, y: x / y)

    def __dot__(self, other):
        """
        Computes the dot product of two arrays.

        Args:
            other (Array): Another Array object 

        Returns:
            Array: A new Array object with the result of the dot product.

        Raises:
            ValueError: If the shapes of the arrays are not compatible for the dot product.
        """

        # Dot product of two 1D arrays
        if len(self.shape) == len(other.shape) == 1 and self.shape[0] == other.shape[0]:
            return sum(self.data[i] * other.data[i] for i in range(self.shape[0]))
        
        # Dot product of two 2D arrays (Matrix multiplication)
        elif (
            len(self.shape) == len(other.shape) == 2 and self.shape[1] == other.shape[0]
        ):
            result = [
                [
                    sum(a * b for a, b in zip(self_row, other_col))
                    for other_col in zip(*other.data)
                ]
                for self_row in self.data
            ]
            return Array(result)

        # Dot product of a 2D array and a 1D array (Matrix-vector multiplication)
        elif (
            len(self.shape) == 2
            and len(other.shape) == 1
            and self.shape[1] == other.shape[0]
        ):
            result = [
                sum(a * b for a, b in zip(self_row, other.data))
                for self_row in self.data
            ]
            return Array(result)
        else:
            raise ValueError(
                f"Dot product not supported for these shapes {self.shape} {other.shape}"
            )

    def __matmul__(self, other):
        """
        Implements the matrix multiplication using the @ operator.

        Args:
            other (Array): Another Array object.

        Returns:
            Array: A new Array object with the result of the matrix multiplication.

        Raises:
            ValueError: If the shapes of the arrays are not compatible for matrix multiplication.
        """
        return self.__dot__(other)

    def sum(self, axis: Optional[int] = None):
        """
        Computes the sum of array elements over a given axis.

        Args:
            axis (int, optional): The axis along which to sum the elements.
                                  If None, sums all elements of the array.

        Returns:
            int/float/Array: The sum of the elements.
        
        Notes:
            - If axis is 0, computes the sum along rows.
            - If axis is 1, computes the sum along columns.
            - If axis is None or any other value, computes the sum of the flattened data.
        """

        # Sum all elements
        if axis is None:
            if len(self.shape) == 1: 
                return sum(self.data)
            elif len(self.shape) == 2:
                return sum(sum(item) for item in self.data)
        
        # Sum along the first axis (columns)
        elif axis == 0:
            if len(self.shape) == 1:
                return sum(self.data)
            elif len(self.shape) == 2:    
                return Array(
                    [sum(row[i] for row in self.data) for i in range(self.shape[1])]
                )
        
        # Sum along the second axis (rows)
        elif axis == 1:
            if len(self.shape) != 2:
                raise ValueError(f"Axis 1 is not valid for array with shape {self.shape}")
            return Array([sum(row) for row in self.data])
        
        else:
            raise ValueError(f"Invalid axis")
    
    def __pow__(self, power):
        """
        Raises each element of the array to the specified power.

        Args:
            power (float): The power to which each element is raised.

        Returns:
            Array: A new Array object with each element raised to the specified power.
        """
        if len(self.shape) == 1:
            return Array([x ** power for x in self.data])
        elif len(self.shape) == 2:
            return Array([[x ** power for x in row] for row in self.data])
        else:
            raise ValueError("__pow__ method is only implemented for 1D and 2D arrays.")
        
    def sqrt(self):
        """
        Computes the square root of each element in the array.

        Returns:
            Array: A new Array object with the square root of each element.
        """
        if len(self.shape) == 1:
            return Array([math.sqrt(x) for x in self.data])
        elif len(self.shape) == 2:
            return Array([[math.sqrt(x) for x in row] for row in self.data])
        else:
            raise ValueError("sqrt method is only implemented for 1D and 2D arrays.")

