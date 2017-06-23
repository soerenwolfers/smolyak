'''
https://stackoverflow.com/questions/1724693/find-a-file-in-python
'''
import os
import fnmatch
def find_files(pattern, path=None):
    if not path:
        path = os.getcwd()
    result = []
    for root, __, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(os.path.join(root,name),pattern):
                result.append(os.path.join(root, name))
    return result

def find_directories(pattern, path=None):
    if not path:
        path = os.getcwd()
    result = []
    for root, __, __ in os.walk(path):
        if fnmatch.fnmatch(root,pattern):
            result.append(root)
    return result
