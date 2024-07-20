import copy
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
    
    def __neg__(self):
        """
        Negates each element of the Array.

        Returns:
            Array: A new Array object with each element negated.
        """
        if len(self.shape) == 1:
            negated_data = [-x for x in self.data]
        else:
            negated_data = [[-x for x in row] for row in self.data]
        
        return Array(negated_data)

    def getitem(self, index):
        """
        Gets the element at the specified index.

        Args:
            index (int or tuple): The index of the element to get.

        Returns:
            int, float, or list: The element at the specified index.
        """
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value

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
    
    def copy(self):
        """
        Creates a deep copy of the Array object.

        Returns:
            Array: A new Array object that is a deep copy of the original.
        """
        return Array(copy.deepcopy(self.data))
    
    @staticmethod
    def ensure_array(obj):
        """
        Ensures that the input is an Array instance. If the input is a list, it converts it to an Array.

        Args:
            obj (list or Array): The input object.

        Returns:
            Array: The Array instance.
        """
        if isinstance(obj, Array):
            return obj
        elif isinstance(obj, list):
            return Array.from_list(obj)
        else:
            raise ValueError("Input must be a list or an Array instance")

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

    def _elementwise_op(self, other, op, reverse_op=None):
        """
        Helper method to perform element-wise operations.

        Args:
            other (Array or int or float): Another Array object or a scalar.
            op (callable): A function to perform the operation.
            reverse_op (callable, optional): A function to perform the reverse operation if the first operand is a scalar.

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
    
    def __radd__(self, other):
        return self._elementwise_op(other, lambda x, y: y + x)

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
    
    def __rsub__(self, other):
        return self._elementwise_op(other, lambda x, y: y - x)

    def __mul__(self, other):
        """
        Multiplies two arrays element-wise or multiplies each element of the array by a scalar.

        Args:
            other (Array or int or float): Another Array object or a scalar.

        Returns:
            Array: A new Array object with the result of the multiplication.
        """
        return self._elementwise_op(other, lambda x, y: x * y)
    
    def __rmul__(self, other):
        return self._elementwise_op(other, lambda x, y: y * x)

    def __truediv__(self, other):
        """
        Divides one array by another element-wise or divides each element of the array by a scalar.

        Args:
            other (Array or int or float): Another Array object or a scalar.

        Returns:
            Array: A new Array object with the result of the division.
        """
        return self._elementwise_op(other, lambda x, y: x / y)
    
    def __rtruediv__(self, other):
        return self._elementwise_op(other, lambda x, y: y / x)

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

    @staticmethod
    def sum(data, axis: Optional[int] = None):
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
        data = Array.ensure_array(data)
        # Sum all elements
        if axis is None:
            if len(data.shape) == 1: 
                return sum(data.data)
            elif len(data.shape) == 2:
                return sum(sum(item) for item in data.data)
        
        # Sum along the first axis (columns)
        elif axis == 0:
            if len(data.shape) == 1:
                return sum(data.data)
            elif len(data.shape) == 2:    
                return Array(
                    [sum(row[i] for row in data.data) for i in range(data.shape[1])]
                )
        
        # Sum along the second axis (rows)
        elif axis == 1:
            if len(data.shape) != 2:
                raise ValueError(f"Axis 1 is not valid for array with shape {data.shape}")
            return Array([sum(row) for row in data.data])
        
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

    @staticmethod 
    def sqrt(data):
        """
        Computes the square root of each element in the array.

        Returns:
            Array: A new Array object with the square root of each element.
        """
        data = Array.ensure_array(data)
        if len(data.shape) == 1:
            for x in data.data:
                if x < 0:
                    raise ValueError("Cannot compute square root of negative number")
            return Array([math.sqrt(x) for x in data.data])
        else:
            for row in data.data:
                for x in row:
                    if x < 0:
                        raise ValueError("Cannot compute square root of negative number")
            return Array([[math.sqrt(x) for x in row] for row in data.data])
    
    @staticmethod
    def log(data):
        """
        Computes the natural logarithm (base e) of each element in the array.

        Returns:
            Array: A new Array object with the natural logarithm of each element.
        
        Raises:
            ValueError: If the array contains non-positive values.
        """
        data = Array.ensure_array(data)
        if len(data.shape) == 1:
            if any(x <= 0 for x in data.data):
                raise ValueError("log method only supports positive values.")
            return Array([math.log(x) for x in data.data])
        elif len(data.shape) == 2:
            if any(any(x <= 0 for x in row) for row in data.data):
                raise ValueError("log method only supports positive values.")
            return Array([[math.log(x) for x in row] for row in data.data])
        else:
            raise ValueError("log method is only implemented for 1D and 2D arrays.")
    
    @staticmethod
    def abs(data):
        data = Array.ensure_array(data)
        if len(data.shape) == 1:
            return Array([x if x>=0 else -(x) for x in data.data])
        else:
            return Array([[x if x>=0 else -(x) for x in row] for row in data.data])
    
    @staticmethod
    def exp(data):
        data = Array.ensure_array(data)
        if len(data.shape) == 1:
            return Array([math.exp(x) for x in data.data])
        else:
            return Array([[math.exp(x) for x in row] for row in data.data])  

        


