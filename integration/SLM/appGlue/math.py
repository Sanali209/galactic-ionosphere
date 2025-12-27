def Remap(value, min1, max1, min2, max2):
    '''Remap a value from one range to another.
    Example: Remap(25, 0, 100, 0, 1) == 0.25'''
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1)


def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value
