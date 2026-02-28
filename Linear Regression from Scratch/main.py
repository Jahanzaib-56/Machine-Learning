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

''' Implementing Gradient Descent '''
def compute_gradient_descent(df, initial_b, initial_m, learningRate, num_iteration):
    
    b = initial_b
    m = initial_m

    for i in range(num_iteration):

        b, m = step_gradient(b, m, np.array(df), learningRate)

    return [b, m]
