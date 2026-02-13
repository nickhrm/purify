import numpy as np
from purify.utils.purification_util import Purification
from purify.entanglement import Entanglement
from purify.my_time import Time
from purify.constants_tuple import ConstantsTuple, LambdaSrategy

# Constants
t_c = 0.01
lambdas = (0.3, 0.0, 0.0)
F_fresh = 1.0 - sum(lambdas) # 0.7
delta_t = 0.0001  # 20000 / 2e8
p_g = 0.3047
avg_steps_for_link = 1 / p_g
avg_wait_time = avg_steps_for_link * delta_t # ~ 0.33ms

print(f"Fresh Fidelity: {F_fresh}")
print(f"Avg Wait Time for Link: {avg_wait_time:.6f} s")

# Decay calculation
def decay(f_start, duration):
    # (exp(-t/tc) * (f - 0.25) + 0.25)
    return np.exp(-duration / t_c) * (f_start - 0.25) + 0.25

# Mock Entanglement object for util calls
class MockEntanglement:
    def __init__(self, f, l2=0, l3=0):
        self.f = f
        self.l2 = l2
        self.l3 = l3
    def get_current_fidelity(self): return self.f
    def get_current_lambda_2(self): return self.l2
    def get_current_lambda_3(self): return self.l3

# Scenario: We have a link with fidelity F_curr (decayed a bit from 0.7 or boosted)
# A new link arrives (F_fresh).

# Strategy 1: Replace
# Value = F_fresh (immediate)
# (Though strictly, we have to wait for request. But comparing immediate potential is a good proxy)

# Strategy 2: PMD
# We use F_curr and F_fresh.
# Success prob p.
# F_new on success.
# Failure: Loss. Value = Expected value of waiting for NEXT fresh link?
# Value_fail = F_fresh * decay(avg_wait_time)

# Let's iterate various F_curr
f_values = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

print("\n--- Comparison ---")
for f_curr in f_values:
    # 1. Decay f_curr by avg_wait_time (assuming we held it while waiting for this new link)
    # Actually, f_curr IS the fidelity at the moment the new link arrives.
    
    # Value Replace
    val_replace = F_fresh
    
    # Value PMD
    e_good = MockEntanglement(f_curr)
    e_bad = MockEntanglement(F_fresh) # Fresh link has l2=l3=0
    
    p_success = Purification._pmd_success_probability(e_good, e_bad)
    if p_success > 0:
        f_pumped = Purification._pmd_jump_function(e_good, e_bad)
    else:
        f_pumped = 0
        
    # Value if fail: We start from scratch. We have NOTHING.
    # We must wait for next link.
    # The next link will have F_fresh, but we have to wait avg_wait_time.
    # Discounting? PPO with gamma=1 implies no discounting?
    # But Qubit quality decays!
    # Value ~ F_fresh * Qubit_Decay(avg_wait_time).
    # Qubit Decay: coherence time t_c = 0.01.
    qubit_factor = decay(1.0, avg_wait_time) # Assuming qubit starts at 1.0 (wait time 0 -> fidelity 1)
    
    val_fail = F_fresh * qubit_factor
    
    # Expected Value PMD (immediate fidelity potential, discounted by risk)
    # Actually, we should compare "Effective Fidelity"
    # If PMD succeeds: We have f_pumped. We hold it.
    # If PMD fails: We have F_fresh * risk_factor.
    
    val_pmd = p_success * f_pumped + (1 - p_success) * val_fail
    
    print(f"F_curr: {f_curr:.4f}")
    print(f"  Replace: {val_replace:.4f}")
    print(f"  PMD:     {val_pmd:.4f} (p={p_success:.2f}, F_up={f_pumped:.4f}, Val_fail={val_fail:.4f})")
    print(f"  Decision: {'PMD' if val_pmd > val_replace else 'REPLACE'}")

    
