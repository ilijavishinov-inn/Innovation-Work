import re
from typing import List


def sortl_list_of_strings_alphanumerically(list_of_strings: List[str]) -> list:

    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    list_of_strings.sort(key=alphanum, reverse = True)
    return list_of_strings



































