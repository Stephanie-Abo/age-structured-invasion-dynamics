import numpy as np
import numba as nb
from numba import prange, config
from scipy.linalg import solve_banded
from scipy.integrate import simpson
from collections import deque

@nb.jit(parallel=True, fastmath=True, cache=True)
def compute_death_rate_nb(death_case, mu0, gamma, alpha, a, P, mu):
    na, nx = mu.shape
    if death_case == 1:
        for i in prange(na):
            for j in range(nx):
                mu[i, j] = mu0
    elif death_case == 2:
        for i in prange(na):
            for j in range(nx):
                mu[i, j] = mu0 * P[j]
    elif death_case == 3:
        for i in prange(na):
            for j in range(nx):
                mu[i, j] = (mu0 - gamma * a[i] * np.exp(-alpha * a[i])) * P[j]
    
@nb.jit(parallel=True, fastmath=True, cache=True)
def compute_birth_rate_nb(birth_case, b0, alpha, a, P, beta):
    na = len(a)
    nx = len(P)

    for i in prange(na):
        for j in range(nx):
            if birth_case == 1:
                beta[i, j] = b0 * (1 - P[j])
            elif birth_case == 2:
                beta[i, j] = b0 * np.exp(-alpha * a[i]) * (1 - P[j])
            elif birth_case == 3:
                beta[i, j] = b0 * a[i] * np.exp(-alpha * a[i]) * (1 - P[j])

class DiffusionSolver_numba:
    def __init__(self, alpha, mu0, b0, gamma, T, a_max, kappa, dx, L, da, birth_case, death_case, save_every=None):
        # Physical params
        self.mu0 = mu0
        self.b0 = b0
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = kappa
        self.birth_case = birth_case
        self.death_case = death_case
        
        # Discretisation
        self.T = T
        self.a_max = a_max
        self.dx = dx
        self.L = L
    
        # We implemented Crank–Nicolson, which is unconditionally stable for diffusion
        # So no longer limited by CFL in the same way
        self.da = da 
        self.dt = self.da  # still match da for characteristic marching

        # Grids
        self.nt = int(np.ceil(self.T / self.dt)) + 1

        # spatial grid: ensure full coverage of [-L, L]
        self.nx = int(np.ceil(2*self.L / self.dx)) + 1
        self.x  = np.linspace(-self.L, self.L, self.nx)
        self.dx = self.x[1] - self.x[0]

        # age grid: cover [0, a_max + T]
        self.na = int(np.ceil(self.a_max / self.da)) + 1
        self.max_a = self.a_max + self.T                         # Maximum age 
        self.max_na = int(np.ceil(self.max_a / self.da)) + 1     # Maximum possible age steps
        self.age_grid = np.linspace(0, self.max_a, self.max_na)

        # Pre-allocate working arrays 
        self.mu_array = np.zeros((self.max_na, self.nx))
        self.beta_array = np.zeros((self.max_na, self.nx))
        self.birth_integrand = np.zeros((self.max_na, self.nx))

        self.u_next = np.zeros((self.max_na, self.nx))
        self.u = np.zeros((self.max_na, self.nx))
        self.exp_a = np.zeros(self.max_na)
        self.age_exp_a = np.zeros(self.max_na)

        # Banded matrix representation of the Crank–Nicolson left-hand side operator
        self.ab = np.zeros((3, self.nx))
        dx2 = self.dx ** 2
        kappa_dt = self.kappa * self.dt / dx2

        self.ab[0, 1:] = -0.5 * kappa_dt           # upper diag
        self.ab[1, :]  = 1.0 + kappa_dt            # main diag
        self.ab[2, :-1] = -0.5 * kappa_dt          # lower diag
         
        self.ab[0, 1] = -kappa_dt                  # Neumann BC
        self.ab[2, -2] = -kappa_dt

    def compute_death_rate(self, a, P):
        mu = self.mu_array[:len(a)]
        compute_death_rate_nb(self.death_case, self.mu0, self.gamma, self.alpha, a, P, mu)
        return mu

    def compute_birth_rate(self, a, P):
        beta = self.beta_array[:len(a)]
        compute_birth_rate_nb(self.birth_case, self.b0, self.alpha, a, P, beta)
        return beta

    def compute_integrals(self, a, u):
        if len(a) < 2:
            return np.zeros(self.nx), np.zeros(self.nx), np.zeros(self.nx)

        P = simpson(u, dx=self.da, axis=0) # Total population
        self.exp_a[:len(a)] = np.exp(-self.alpha * a)
        self.age_exp_a[:len(a)] = a * self.exp_a[:len(a)]
        C = simpson(u * self.exp_a[:len(a), None], dx=self.da, axis=0)     # Exponential-weighted total
        D = simpson(u * self.age_exp_a[:len(a), None], dx=self.da, axis=0) # Age-weighted total
        return P, C, D
    
    def reaction_update(self, u, mu, beta):
        """
        Advance u by one ageing step (explicit Euler)
        """
        return u + self.dt * (-mu * u - beta * u)
    
    def diffusion_update(self, u):
        u_new = np.empty_like(u)
        rhs = np.empty_like(u)

        dt = self.dt
        dx2 = self.dx ** 2
        kappa = self.kappa
        factor = 0.5 * dt * kappa / dx2

        # Build RHS
        for i in range(u.shape[0]):
            u_row = u[i]
            rhs[i, 1:-1] = u_row[1:-1] + factor * (u_row[0:-2] - 2*u_row[1:-1] + u_row[2:])
            rhs[i, 0]    = u_row[0]    + factor * (2*u_row[1] - 2*u_row[0])
            rhs[i, -1]   = u_row[-1]   + factor * (2*u_row[-2] - 2*u_row[-1])

        # Solve banded system for each age slice
        for i in range(u.shape[0]):
            u_new[i] = solve_banded((1, 1), self.ab, rhs[i])

        return u_new

    def set_initial_condition(self, a):
        A, X = np.meshgrid(a, self.x, indexing="ij")
    
        ic_width = 2 # fixed width
        self.u = 0.01 * (X < (self.x[0] + ic_width)) * np.exp(-A)
        
    def solve(self, save_every=None, steady_state_tol=1e-3, extend_after_ss=100.0):
        """
        Parameters:
        - save_every: how often to save solutions
        - steady_state_tol: threshold for steady wave detection
        - extend_after_ss: keep running this many time units after first steady-state detection
        """
        self.tail_frac_tol = 1e-6

        # Time stepping allocation
        if save_every is None:
            save_every = max(1, int(1 / self.da))
        
        # Initialize steady state detection
        check_every = 10
        window_size = 6
        passes_needed = 5
        pass_window = deque(maxlen=window_size)
        
        max_nt = int(np.ceil((self.T + extend_after_ss) / self.dt)) + 1
        nsaves = (max_nt // save_every) + 1

        # Preallocate save buffers
        a_list = [None] * nsaves
        u_list = [None] * nsaves
        u0_list = [None] * nsaves
        P_list = [None] * nsaves
        C_list = [None] * nsaves
        D_list = [None] * nsaves

        # Initialise the state
        a = self.age_grid[:self.na]
        self.set_initial_condition(a) 
        u = self.u

        # Compute initial integrals
        P, C, D = self.compute_integrals(a, u)

        # Store t = 0
        save_idx = 0
        a_list[save_idx] = a.copy()
        u_list[save_idx] = u.copy()
        u0_list[save_idx] = u[0].copy()
        P_list[save_idx] = P
        C_list[save_idx] = C
        D_list[save_idx] = D
        save_idx += 1

        # Steady state tracking
        self.t_ss = None
        steady_state_detected = False
        target_end = self.T  # Initial target

        # Main time loop 
        for n in range(1, max_nt + 1):
            t = n * self.dt

            # Store previous state for steady state detection
            u_prev = u.copy()
            
            # ========== PREDICTOR STEP ==========
            # Current integrals from u^n
            P, C, D = self.compute_integrals(a, u)
            mu = self.compute_death_rate(a, P)
            beta = self.compute_birth_rate(a, P)
            
            # Predict interior (reaction + diffusion)
            u_pred = self.reaction_update(u, mu, beta)
            u_pred = self.diffusion_update(u_pred)
            
            # ========== CORRECTOR STEP ==========
            # Recompute rates from predicted solution
            P_pred = simpson(u_pred, dx=self.da, axis=0)
            mu_pred = self.compute_death_rate(a, P_pred)
            beta_pred = self.compute_birth_rate(a, P_pred)
            
            # Average rates
            mu_avg = 0.5 * (mu + mu_pred)
            beta_avg = 0.5 * (beta + beta_pred)
            
            # Corrector: apply averaged rates to original u^n
            u_corr = self.reaction_update(u, mu_avg, beta_avg)
            u_corr = self.diffusion_update(u_corr)
            
            # Corrected birth integral (from corrected interior with averaged rates)
            self.birth_integrand[:len(a)] = beta_avg * u_corr
            birth_integral = simpson(self.birth_integrand[:len(a)], dx=self.da, axis=0)
            
            # ========== AGE MARCH ==========
            # Age domain expansion check
            tail_mass = simpson(u[-1, :], self.x) * self.da
            prev_mass = simpson(simpson(u, self.x, axis=1), a)
            tail_frac = tail_mass / max(prev_mass, 1e-12)
            can_expand = (self.na + 1 <= self.max_na) and (tail_frac > self.tail_frac_tol) and (max(a) < 2000)

            # Age-march one step using corrected solution
            if can_expand:
                u_next = self.u_next[:len(a) + 1]
                u_next[:] = 0
                u_next[1:] = u_corr  # Use corrected
            else:
                u_next = self.u_next[:len(a)]
                u_next[1:] = u_corr[:-1]  # Use corrected
                
            u_next[0] = 2 * birth_integral

            if can_expand:
                self.na += 1
                a = self.age_grid[:self.na]

            u = u_next

            # ========== STEADY STATE DETECTION ==========
            if (n % check_every) == 0 and not steady_state_detected:
                min_na = min(u.shape[0], u_prev.shape[0])
                
                # Simple relative change detection as fallback
                change_norm = np.linalg.norm(u[:min_na] - u_prev[:min_na])
                u_norm = np.linalg.norm(u_prev[:min_na])
                relative_change = change_norm / (u_norm + 1e-12)
                
                # Also check population integrals
                P_prev, _, _ = self.compute_integrals(a[:min_na], u_prev[:min_na])
                P_curr, _, _ = self.compute_integrals(a[:min_na], u[:min_na])
                P_change = np.linalg.norm(P_curr - P_prev) / (np.linalg.norm(P_prev) + 1e-12)
                
                # Use the more sensitive detection method
                max_change = max(relative_change, P_change)
                
                if max_change < steady_state_tol:
                    pass_window.append(True)
                else:
                    pass_window.append(False)
                
                # ----------------------------------------------------------------------------------
                # COMMENT OUT THIS BLOCK TO REMOVE THE PRINTS
                if n % 100 == 0:  # Print every 100 checks
                    print(f"t={t:.3f}, rel_change={relative_change:.2e}, P_change={P_change:.2e}, "
                        f"window={list(pass_window)}")
                # ----------------------------------------------------------------------------------

                # Check if we have enough passes
                if len(pass_window) == window_size and sum(pass_window) >= passes_needed:
                    steady_state_detected = True
                    self.t_ss = t
                    target_end = max(t + extend_after_ss, self.T)  # Extend after detection
                    print(f"Steady state detected at t={self.t_ss:.3f}")
                    print(f"Continuing until t={target_end:.3f} for post-SS analysis")

            # Save data
            if (n % save_every) == 0 and save_idx < nsaves:
                a_list[save_idx] = a.copy()
                u_list[save_idx] = u.copy()
                u0_list[save_idx] = u[0].copy()
                P_list[save_idx] = P
                C_list[save_idx] = C
                D_list[save_idx] = D
                save_idx += 1

            # Stop condition
            if t >= target_end:
                if steady_state_detected:
                    print(f"Steady state analysis complete. Final t={t:.3f}")
                else:
                    print(f"Reached maximum time t={t:.3f} without steady state detection")
                break

        # Final mass check
        tail_mass = np.trapezoid(u[-1, :], self.x) * self.da
        total_mass = np.trapezoid(np.trapezoid(u, self.x, axis=1), self.age_grid[:self.na])
        if tail_mass / total_mass > 1e-5:
            print("Warning: non-negligible mass near a_max. Consider increasing a_max.")

        # Trim unused entries
        a_list = a_list[:save_idx]
        u_list = u_list[:save_idx]
        u0_list = u0_list[:save_idx]
        P_list = P_list[:save_idx]
        C_list = C_list[:save_idx]
        D_list = D_list[:save_idx]

        return a_list, u_list, u0_list, P_list, C_list, D_list, save_every