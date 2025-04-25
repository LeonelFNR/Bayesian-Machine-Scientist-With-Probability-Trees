import inspect
import re
import numpy as np
import os

def get_function_expression(func):
    #get the code of the function
    source = inspect.getsource(func)

    #extract what is returned from the "return" in the original function
    match = re.search(r'return (.+)', source)
    if match:
        expression = match.group(1)
        return expression
    return None

def sanitize_filename(filename):
    #replace invalid characters in the filename
    filename = filename.replace('np.', '')
    filename = re.sub(r'[^a-zA-Z0-9\s\-_.()+*/]+', '', filename)
    # filename = filename.replace('','')
    return filename

# Function that creates the filename from the function, and the parameter and variable file, and saves it in a specified folder
def file_namer(func, parameter_path, folder_path, mean, stddev):
    # func is the function and #parameter_path is the path of the parameter file
    #get the function expression
    expression = get_function_expression(func)
    if expression:
        #ix the expression format
        filename = sanitize_filename(expression)
        # we select the parameter filename from the path
        priorname = os.path.basename(parameter_path)
        #separate the .dat extension from the file name
        name, _ = os.path.splitext(priorname)
        # since the file name format contains several dots, we split it into parts
        match = re.search(r'(\w+)\.(\w+)\.(\w+)\.(\w+)\.(.*)', name)
        #select the part that contains the number of variables and parameters
        paramstring =  match.group(3) + "_" + match.group(4)
        #format the variable for the filename
        filename = filename + "_" + paramstring
        if mean == 0 and stddev == 0:
            filename = filename + ".txt"
        else:
            filename = filename + "_mean" + str(mean) + "_stddev" + str(stddev) + ".txt"
        #save in the specified folder
        filepath = os.path.join(folder_path, filename)
        #return the file path to be able to write to it
        print("File created at " + filepath)
        return filepath
    else:
        print("Could not extract the function expression")
        return None

        