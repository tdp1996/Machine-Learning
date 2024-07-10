from typing import Union
def transpose(a :list[list[Union[float,int]]]) ->list[list[Union[float,int]]]:
    converted_data = [[item[i] for item in a] for i in range(len(a[0]))]
    return converted_data