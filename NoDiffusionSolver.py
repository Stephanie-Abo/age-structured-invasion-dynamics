import numpy as np
from scipy.integrate import simpson
from collections import deque

class NoDiffusionSolver:
    """Non-spatial solver without diffusion using predictor-corrector scheme"""
    def __init__(self, alpha, mu0, b0, gamma, T, a_max, da, birth_case, death_case):
        # Parameters 
        self.alpha = alpha
        self.mu0 = mu0
        self.b0 = b0
        self.gamma = gamma
        self.T = T
        self.a_max = a_max
        self.da = da
        self.birth_case = birth_case
        self.death_case = death_case

        # Discretization 
        self.dt = da  # Match da for characteristic marching
        self.nt = int(T / self.dt)
        self.na = int(a_max / da)
        self.a_max_total = a_max + T  # Maximum possible age
        self.na_total = int(self.a_max_total / da)
        self.age_grid = np.linspace(0, self.a_max_total, self.na_total)
        
        # Tail fraction tolerance for adaptive age grid
        self.tail_frac_tol = 1e-6

    def compute_death_rate(self, a, P):
        """Death rate"""
        if self.death_case == 1:
            return self.mu0 * np.ones_like(a)
        elif self.death_case == 2:
            return self.mu0 * P * np.ones_like(a)
        elif self.death_case == 3:
            return (self.mu0 - self.gamma * a * np.exp(-self.alpha * a)) * P

    def compute_birth_rate(self, a, P):
        """Birth rate"""
        if self.birth_case == 1:
            return self.b0 * (1 - P) * np.ones_like(a)
        elif self.birth_case == 2:
            return self.b0 * np.exp(-self.alpha * a) * (1 - P)
        elif self.birth_case == 3:
            return self.b0 * a * np.exp(-self.alpha * a) * (1 - P)

    def compute_integrals(self, a, u):
        """Compute population integrals P, C, D"""
        P = simpson(u, dx=self.da)  # Total population
        C = simpson(np.exp(-self.alpha * a) * u, dx=self.da)  # Exponential-weighted
        D = simpson(a * np.exp(-self.alpha * a) * u, dx=self.da)  # Age-weighted
        return P, C, D
    
    def reaction_update(self, u, mu, beta):
        """Explicit Euler"""
        return u + self.dt * (-mu * u - beta * u)
    
    def solve(self, save_every=None, steady_state_tol=1e-3, extend_after_ss=100.0):
        """
        Parameters:
        - save_every: how often to save solutions
        - steady_state_tol: threshold for steady wave detection
        - extend_after_ss: keep running this many time units after first steady-state detection
        """
        # Steady-state detection parameters
        check_every = 10
        window_size = 6
        passes_needed = 5
        pass_window = deque(maxlen=window_size)
        
        # Save interval
        if save_every is None:
            save_every = max(1, int(1 / self.da))
        
        max_nt = int(np.ceil((self.T + extend_after_ss) / self.dt)) + 1
        nsaves = (max_nt // save_every) + 1

        # Preallocate save buffers
        a_list = [None] * nsaves
        u_list = [None] * nsaves
        u0_list = [None] * nsaves
        P_list = [None] * nsaves
        C_list = [None] * nsaves
        D_list = [None] * nsaves

        # Initialize state
        a = self.age_grid[:self.na]
        u = np.exp(-10 * a**2)  # Initial condition
        
        # Compute initial integrals
        P, C, D = self.compute_integrals(a, u)

        # Store t = 0
        save_idx = 0
        a_list[save_idx] = a.copy()
        u_list[save_idx] = u.copy()
        u0_list[save_idx] = u[0]
        P_list[save_idx] = P
        C_list[save_idx] = C
        D_list[save_idx] = D
        save_idx += 1

        # Steady-state tracking
        self.t_ss = None
        steady_state_detected = False
        target_end = self.T

        # Main time loop 
        for n in range(1, max_nt + 1):
            t = n * self.dt

            # Store previous state
            u_prev = u.copy()
            
            # ========== PREDICTOR STEP ==========
            # Current integrals from u^n
            P, C, D = self.compute_integrals(a, u)
            mu = self.compute_death_rate(a, P)
            beta = self.compute_birth_rate(a, P)
            
            # Predict interior
            u_pred = self.reaction_update(u, mu, beta)
            
            # ========== CORRECTOR STEP ==========
            # Recompute rates from predicted solution
            P_pred = simpson(u_pred, dx=self.da)
            mu_pred = self.compute_death_rate(a, P_pred)
            beta_pred = self.compute_birth_rate(a, P_pred)
            
            # Average rates
            mu_avg = 0.5 * (mu + mu_pred)
            beta_avg = 0.5 * (beta + beta_pred)
            
            # Corrector: apply averaged rates to original u^n
            u_corr = self.reaction_update(u, mu_avg, beta_avg)
            
            # Corrected birth integral
            birth_integral = simpson(beta_avg * u_corr, dx=self.da)
            
            # ========== AGE MARCH ==========
            # Check if age domain needs expansion
            tail_mass = u[-1] * self.da
            prev_mass = simpson(u, dx=self.da)
            tail_frac = tail_mass / max(prev_mass, 1e-12)
            can_expand = (self.na + 1 <= self.na_total) and (tail_frac > self.tail_frac_tol) and (max(a) < 2000)

            # Age-march one step using corrected solution
            if can_expand:
                u_next = np.zeros(len(u) + 1)
                u_next[1:] = u_corr
            else:
                u_next = np.zeros(len(u))
                u_next[1:] = u_corr[:-1]  # Drop oldest
                
            u_next[0] = 2 * birth_integral

            # Extend age grid if allowed
            if can_expand:
                self.na += 1
                a = self.age_grid[:self.na]

            u = u_next

            # ========== STEADY STATE DETECTION ==========
            if (n % check_every) == 0 and not steady_state_detected:
                min_na = min(u.shape[0], u_prev.shape[0])
                
                # Relative change in solution
                change_norm = np.linalg.norm(u[:min_na] - u_prev[:min_na])
                u_norm = np.linalg.norm(u_prev[:min_na])
                relative_change = change_norm / (u_norm + 1e-12)
                
                # Also check population integral
                P_prev = simpson(u_prev[:min_na], dx=self.da)
                P_curr = simpson(u[:min_na], dx=self.da)
                P_change = abs(P_curr - P_prev) / (P_prev + 1e-12)
                
                # Use the more sensitive detection method
                max_change = max(relative_change, P_change)
                
                if max_change < steady_state_tol:
                    pass_window.append(True)
                else:
                    pass_window.append(False)
                
                # Check if we have enough consecutive passes
                if len(pass_window) == window_size and sum(pass_window) >= passes_needed:
                    steady_state_detected = True
                    self.t_ss = t
                    target_end = max(t + extend_after_ss, self.T)
                    print(f"[Non-spatial] Steady state detected at t={self.t_ss:.3f}")
                    print(f"[Non-spatial] Continuing until t={target_end:.3f}")

            # Save data
            if (n % save_every) == 0 and save_idx < nsaves:
                a_list[save_idx] = a.copy()
                u_list[save_idx] = u.copy()
                u0_list[save_idx] = u[0]
                P_list[save_idx] = P
                C_list[save_idx] = C
                D_list[save_idx] = D
                save_idx += 1

            # Stop condition
            if t >= target_end:
                if steady_state_detected:
                    print(f"[Non-spatial] Complete at t={t:.3f}")
                else:
                    print(f"[Non-spatial] Reached maximum time t={t:.3f} without steady state")
                break

        # Final tail mass check
        tail_mass = u[-1] * self.da
        total_mass = simpson(u, dx=self.da)
        if tail_mass / total_mass > 1e-5:
            print("[Non-spatial] Warning: non-negligible mass near a_max")

        # Trim unused entries
        a_list = a_list[:save_idx]
        u_list = u_list[:save_idx]
        u0_list = u0_list[:save_idx]
        P_list = P_list[:save_idx]
        C_list = C_list[:save_idx]
        D_list = D_list[:save_idx]

        return a_list, u_list, u0_list, P_list, C_list, D_list