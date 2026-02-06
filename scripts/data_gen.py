import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

# Constants
MU = 398600.44  # Earth's gravitational parameter, km^3/s^2
EARTH_RADIUS = 6378.137  # km

def orbit_equations(t, state):
    """
    Standard Two-Body Problem equations of motion.
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Gravitational acceleration
    ax = -MU * x / r**3
    ay = -MU * y / r**3
    az = -MU * z / r**3
    
    return [vx, vy, vz, ax, ay, az]

def generate_orbit(initial_state, t_span, t_eval, anomaly_type=None, anomaly_time=None):
    """
    Generate orbit data with optional anomalies.
    """
    def event_anomaly(t, state):
        return t - anomaly_time
    
    sol = solve_ivp(
        orbit_equations, 
        t_span, 
        initial_state, 
        t_eval=t_eval, 
        rtol=1e-9, 
        atol=1e-12
    )
    
    data = pd.DataFrame(sol.y.T, columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
    data['t'] = sol.t
    
    if anomaly_type == 'maneuver' and anomaly_time is not None:
        # Simulate a sudden Delta-V (e.g., station keeping or collision avoidance)
        idx = np.searchsorted(sol.t, anomaly_time)
        if idx < len(data):
            # Add 0.1 km/s to velocity at anomaly time
            data.loc[idx:, 'vx'] += 0.05
            data.loc[idx:, 'vy'] += 0.05
            # We would ideally re-propagate from here, but for a simple anomaly
            # detection demo, a step change or a slight shift is often used.
            # For higher realism, we'd restart the solver from this point.
            
    elif anomaly_type == 'sensor_noise' and anomaly_time is not None:
        idx = np.searchsorted(sol.t, anomaly_time)
        # Add significant noise to a segment
        noise_mask = (data['t'] >= anomaly_time) & (data['t'] < anomaly_time + 1000)
        data.loc[noise_mask, ['x', 'y', 'z']] += np.random.normal(0, 5, size=(noise_mask.sum(), 3))

    return data

def main():
    # Example: ISS-like orbit (approx 400km altitude)
    # R_iss approx 6778 km, V_iss approx 7.66 km/s
    altitude = 400
    r0 = EARTH_RADIUS + altitude
    v0 = np.sqrt(MU / r0)
    
    initial_state = [r0, 0, 0, 0, v0, 0]
    duration = 24 * 3600 # 24 hours
    t_span = (0, duration)
    t_eval = np.linspace(0, duration, 1000) # 1000 points
    
    print("Generating Nominal Orbit...")
    nominal_data = generate_orbit(initial_state, t_span, t_eval)
    
    print("Generating Anomaly Orbit (Maneuver)...")
    anomaly_data = generate_orbit(initial_state, t_span, t_eval, anomaly_type='maneuver', anomaly_time=12*3600)
    
    # Save datasets
    os.makedirs('data', exist_ok=True)
    nominal_data.to_csv('data/orbit_nominal.csv', index=False)
    anomaly_data.to_csv('data/orbit_anomaly.csv', index=False)
    print("Data saved to data/ folder.")

if __name__ == "__main__":
    main()
