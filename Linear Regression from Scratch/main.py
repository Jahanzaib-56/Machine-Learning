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

def step_gradient(curr_b, curr_m, df, learningRate):

    # Starting point for our gradient
    b_gradient = 0
    m_gradient = 0

    N = float(len(df))

    # Iterating over each point
    for i in range(0, len(df)):

        X = df[i, 0]
        Y = df[i, 1]

        # Computing partial derivatives of our error function
        b_gradient += -(2/N) * (Y - ((curr_m * X) + curr_b))
        m_gradient += -(2/N) * X * (Y - ((curr_m * X) + curr_b))

    # Updating values of b  and m using partial derivatives.
    new_b = curr_b - (learningRate * b_gradient)
    new_m = curr_m - (learningRate * m_gradient)

    return [new_b, new_m]

def LinearRegression():

    # Step-01: Load the data
    df = np.genfromtxt('data.csv', delimiter=',')

    # Step-02: Defining hyperparameters

    # Learning rate: It dictates how fast our model should converge i.e get the best fit line
    alpha = 0.0001 

    # paramters of line equation y = mx + b
    initial_m = 0
    initial_b = 0


    num_iteration = 1000 # Our model will be trained 1000 times

    print(f'Starting grad_descent at b = {initial_b}, m = {initial_m}, error = {compute_error(initial_b, initial_m, df)}')

    [b, m] = compute_gradient_descent(df, initial_b, initial_m, alpha, num_iteration)

    print("\nRunning.........")

    print(f'\nAfter {num_iteration} iterations, b = {b}, m = {m}, error = {compute_error(b, m, df)} ')

if __name__ == '__main__':
    LinearRegression()
