import numpy as np
from scipy.integrate import simpson

def mean_age_at_division(CCTD, ages):
    """
    Compute the mean age at division using the cell cycle time distribution f(a)=CTTD(a)
    """
    assert callable(CCTD), "Input 'f' must be a callable function of age"
    assert isinstance(ages, np.ndarray), "'ages' must be a NumPy array"

    return simpson(ages * CCTD(ages), ages)

def mean_population_age(SSAD, ages):
    """
    Compute the mean age of the population using the steady state age distribution F(a)=SSAD(a)
    """
    assert callable(SSAD), "Input 'F' must be a callable function of age"
    assert isinstance(ages, np.ndarray), "'ages' must be a NumPy array"

    values = SSAD(ages)
    numerator = simpson(ages * values, ages)
    denominator = simpson(values, ages)

    if denominator == 0:
        raise ValueError("Denominator is zero; possibly empty population")
    
    return numerator / denominator


def division_rate(beta, alpha, ages, P_bar, case):
    """
    Compute the age-dependent division rate based on case
    """
    assert isinstance(beta, (int, float)), "'beta' must be a number"
    assert isinstance(alpha, (int, float)), "'alpha' must be a number"
    assert isinstance(P_bar, (int, float)), "'P_bar' must be a number"
    assert isinstance(ages, np.ndarray), "'ages' must be a NumPy array"
    assert case in [0, 1, 2, 3, 4, 5, 6], "Invalid case. Must be 0, 1, 2, 3, 4, 5 or 6"

    functions = {
        0: lambda: beta,
        1: lambda: beta * (1 - P_bar),
        2: lambda: beta * np.exp(-alpha * ages) * (1 - P_bar),
        3: lambda: beta * ages * np.exp(-alpha * ages) * (1 - P_bar), # death rate is constant
        4: lambda: beta * ages * np.exp(-alpha * ages) * (1 - P_bar), # death rate depends on P_bar
        5: lambda: beta * ages * np.exp(-alpha * ages) * (1 - P_bar), # death rate depends on age and P_bar
        6: lambda: beta * (1 - P_bar) # death rate depends on P_bar
    }

    return functions[case]()

def boundary_condition(beta, mu, gamma, P_bar, case):
    """
    Compute the boundary condition at age a = 0 based on case
    """
    assert isinstance(beta, (int, float)), "'beta' must be a number"
    assert isinstance(P_bar, (int, float)), "'P_bar' must be a number"
    assert case in [0, 1, 2, 3, 4, 5, 6], "Invalid case. Must be 0, 1, 2, 3, 4, 5 or 6"
    
    # D_bar in case 5 
    D_bar_5 = (mu * P_bar**2) / (gamma * P_bar + beta * (1 - P_bar))
    
    functions = {
        0: lambda: 2 * beta * P_bar,
        1: lambda: 2 * beta * (1 - P_bar) * P_bar,
        2: lambda: 2 * beta * (1 - P_bar) * (mu * P_bar) / (beta * (1 - P_bar)), # C_bar = (mu * P_bar) / (beta * (1 - P_bar))
        3: lambda: 2 * beta * (1 - P_bar) * (mu * P_bar) / (beta * (1 - P_bar)), # D_bar = (mu * P_bar) / (beta * (1 - P_bar))
        4: lambda: 2 * beta * (1 - P_bar) * (mu * P_bar**2) / (beta * (1 - P_bar)), # D_bar = (mu * P_bar**2) / (beta * (1 - P_bar))
        5: lambda: 2 * beta * (1 - P_bar) * D_bar_5,
        6: lambda: 2 * beta * (1 - P_bar) * P_bar
    }

    return functions[case]()

# Define a dictionary of CCTD (cell cycle time distribution) functions
def make_cctd_functions(beta, alpha, mu, gamma, P_bar):
    return {
        0: lambda ages: division_rate(beta, alpha, ages, P_bar, case=0) * np.exp(-2 * beta * ages),
        1: lambda ages: division_rate(beta, alpha, ages, P_bar, case=1) * 
                     np.exp(-(mu + beta * (1 - P_bar)) * ages),
        2: lambda ages: division_rate(beta, alpha, ages, P_bar, case=2) *
                     np.exp(-mu * ages - (beta * (1 - P_bar) / alpha) * (1 - np.exp(-alpha * ages))),
        3: lambda ages: division_rate(beta, alpha, ages, P_bar, case=3) *
                     np.exp(-mu * ages - (beta * (1 - P_bar) / alpha**2) 
                            * (1 - np.exp(-alpha * ages) * (1 + alpha * ages))),
        4: lambda ages: division_rate(beta, alpha, ages, P_bar, case=4)*
                     np.exp(-mu * ages * P_bar - (beta * (1 - P_bar) / alpha**2) 
                            * (1 - np.exp(-alpha * ages) * (1 + alpha * ages))),
        5: lambda ages: division_rate(beta, alpha, ages, P_bar, case=5) *
                     np.exp(-mu * ages * P_bar + ((gamma * P_bar) / alpha**2) 
                            * (1 - np.exp(-alpha * ages) * (1 + alpha * ages))) *
                     np.exp(- (beta * (1 - P_bar) / alpha**2) 
                            * (1 - np.exp(-alpha * ages) * (1 + alpha * ages))),
        6: lambda ages: division_rate(beta, alpha, ages, P_bar, case=6) * 
                     np.exp(-(mu * P_bar + beta * (1 - P_bar)) * ages),
    }

# Define a dictionary of steady-state age distributions
def make_steady_state_distributions(beta, alpha, mu, gamma, P_bar):
    return {
        0: lambda ages: boundary_condition(beta, mu, gamma, P_bar, case=0) * 
                     np.exp(-2 * beta * ages),
        1: lambda ages: boundary_condition(beta, mu, gamma, P_bar, case=1) * 
                     np.exp( -(mu + beta * (1 - P_bar)) * ages),
        2: lambda ages: boundary_condition(beta, mu, gamma, P_bar, case=2) *
                     np.exp(-mu * ages - (beta * (1 - P_bar) / alpha) * (1 - np.exp(-alpha * ages))),
        3: lambda ages: boundary_condition(beta, mu, gamma, P_bar, case=3) *
                     np.exp(-mu * ages - (beta * (1 - P_bar) / alpha**2) *
                            (1 - np.exp(-alpha * ages) * (1 + alpha * ages))),
        4: lambda ages: boundary_condition(beta, mu, gamma, P_bar, case=4) *
                     np.exp(-mu * ages * P_bar - (beta * (1 - P_bar) / alpha**2) *
                            (1 - np.exp(-alpha * ages) * (1 + alpha * ages))),
        5: lambda ages: boundary_condition(beta, mu, gamma, P_bar, case=5) *
                     np.exp(-mu * ages * P_bar +
                            ((gamma * P_bar) / alpha**2) * (1 - np.exp(-alpha * ages) * (1 + alpha * ages))) *
                     np.exp(- (beta * (1 - P_bar) / alpha**2) *
                            (1 - np.exp(-alpha * ages) * (1 + alpha * ages))),
        6: lambda ages: boundary_condition(beta, mu, gamma, P_bar, case=6) * 
                     np.exp( -(mu * P_bar + beta * (1 - P_bar)) * ages)
    }

# Compute mean ages
def compute_mean_ages(beta, alpha, gamma, mu, P_bars, ages, case_num, return_values=False):
    mean_age_division = []
    mean_population_ages = []

    if isinstance(P_bars, float):
        P_bars = [P_bars]

    for P_bar in P_bars:
        # Get the functions for this case
        cctd_func = make_cctd_functions(beta, alpha, mu, gamma, P_bar)[case_num]
        ssd_func = make_steady_state_distributions(beta, alpha, mu, gamma, P_bar)[case_num]

        # Compute the mean ages using the functions
        mean_age_division.append(mean_age_at_division(cctd_func, ages))
        mean_population_ages.append(mean_population_age(ssd_func, ages))
    if return_values:
        return mean_age_division[0], mean_population_ages[0]
    else:
        print("Mean age at division\t" + "\t".join(f"{x:.4f}" for x in mean_age_division))
        print("Mean population age\t" + "\t".join(f"{x:.4f}" for x in mean_population_ages))