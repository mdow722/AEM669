# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def getEigTypes(eigvalue_arr):
    real_neg = 0
    real_pos = 0
    comp_neg = 0
    comp_pos = 0
    for eigval in eigvalue_arr:
        if eigval.real >= 0:
            if eigval.imag == 0:
                real_pos += 1
            else:
                comp_pos += 1
        else:
            if eigval.imag == 0:
                real_neg += 1
            else:
                comp_neg += 1
    return [real_neg,real_pos,comp_neg,comp_pos]