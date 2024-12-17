import math
def exponential_schedule_extended(x, a=1, x_max=1000):
    """
    Extended exponential scheduling function that increases from 0 to 1 in a convex manner
    over an extended range of x from 0 to x_max.
    
    Args:
    x (float): A value between 0 and x_max inclusive.
    a (float): A positive constant to adjust the steepness of the curve.
    x_max (float): The maximum value of x, defaults to 1000.

    Returns:
    float: The output of the adjusted exponential function.
    """
    if 0 <= x <= x_max:
        scaled_x = x / x_max  # Scale x to be between 0 and 1
        return (math.exp(a * scaled_x) - 1) / (math.exp(a) - 1)
    else:
        raise ValueError(f"Input x should be between 0 and {x_max}.")

# Testing the extended function with a few values over the range 0 to 1000
test_values_extended = [0, 250, 500, 750, 1000]
results_exponential_extended = [exponential_schedule_extended(x) for x in test_values_extended]
results_exponential_extended

def concave_schedule(x, a=1, x_max=1000):
    """
    Concave scheduling function that increases from 0 to 1 in a concave manner
    over an extended range of x from 0 to x_max.
    
    Args:
    x (float): A value between 0 and x_max inclusive.
    a (float): A positive constant to adjust the steepness of the curve.
    x_max (float): The maximum value of x, defaults to 1000.

    Returns:
    float: The output of the adjusted concave function.
    """
    if 0 <= x <= x_max:
        scaled_x = x / x_max  # Scale x to be between 0 and 1
        return 1 - math.exp(-a * scaled_x)
    else:
        raise ValueError(f"Input x should be between 0 and {x_max}.")

# Testing the concave function with a few values over the range 0 to 1000
results_concave = [concave_schedule(x) for x in test_values_extended]
results_concave