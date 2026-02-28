import numpy as np

''' Implementing Mean Squared Error '''
def compute_error(b, m, df):
    # initialize it to zero
    totalError = 0
    # Length of our data
    N = len(df)

    # for every point
    for i in range(0, N):
        # Get value of X
        x = df[i, 0]
        # Get value of Y
        y = df[i, 1]

        # Get the Error of a single point
        totalError += (y - (m * x + b)) ** 2

    # Get the Mean Error
    return totalError / float(N)
