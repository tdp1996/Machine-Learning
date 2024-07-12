import math

class Array:
    def __init__(self,data):
        """
        Initializes the Array object with given data.

        Args:
            data (list): A list or list of lists containing the array elements.
        """
        self.data = data
        if isinstance(data[0], list):
            self.shape = (len(data), len(data[0]))
        else:
            self.shape = (len(data),)
        

    def __repr__(self) -> str:
        """
        Returns the string representation of the Array object.

        Returns:
            str: The string representation of the array.
        """
        return (f"Array({self.data})")
    
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
            # other_list = other.tolist()
            if self.shape == other.shape:
                if len(self.shape) == 1:
                    return Array([op(self.data[i], other.data[i]) for i in range(len(self.data))])
                elif len(self.shape) == 2:
                    return Array([[op(self.data[i][j], other.data[i][j]) for j in range(self.shape[1])] for i in range(self.shape[0])])
            elif len(self.shape) == 2 and len(other.shape) == 1 and self.shape[1] == other.shape[0]:
                return Array([[op(self.data[i][j], other.data[j]) for j in range(self.shape[1])] for i in range(self.shape[0])])
            elif len(self.shape) == 1 and len(other.shape) == 2 and self.shape[0] == other.shape[1]:
                return Array([[op(self.data[j], other.data[i][j]) for j in range(other.shape[1])] for i in range(other.shape[0])])
            else:
                raise ValueError(f"operands could not be broadcast together with shapes {self.shape} {other.shape}")
        else:
            if len(self.shape) == 1:
                return Array([op(self.data[i], other) for i in range(len(self.data))])
            else:
                return Array([[op(self.data[i][j], other) for j in range(self.shape[1])] for i in range(self.shape[0])])
    
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
        if len(self.shape) == 1 and len(other.shape) == 1:
            # Dot product of two 1D arrays
            return sum(self.data[i] * other.data[i] for i in range(self.shape[0]))
        elif len(self.shape) == 2 and len(other.shape) == 2:
            # Dot product of two 2D arrays (Matrix multiplication)
            result = [[sum(a * b for a, b in zip(self_row, other_col)) for other_col in zip(*other.data)] for self_row in self.data]
            return Array(result)
        elif len(self.shape) == 2 and len(other.shape) == 1:
            # Dot product of a 2D array and a 1D array (Matrix-vector multiplication)
            result = [sum(a * b for a, b in zip(self_row, other.data)) for self_row in self.data]
            return Array(result)
        else:
            raise ValueError("Dot product not supported for these shapes")
        
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
    

    




