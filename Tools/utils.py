import os
import fnmatch

def find_files(path, extension):  # recursively find files at path with extension; pulled from StackOverflow
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, extension):
            yield os.path.join(root, file)