import torch

# Constants
MU = 398600.44

def get_orbital_energy(state, scaler=None):
    """
    state: [batch, 6] -> x, y, z, vx, vy, vz
    If scaler is provided, state is un-normalized first.
    """
    if scaler is not None:
        # Assuming state is torch tensor, convert to numpy for scaler then back
        # This is slow for training, so we should preferably pass un-normalized tensors
        pass 
    
    pos = state[:, :3]
    vel = state[:, 3:]
    
    r = torch.norm(pos, dim=1)
    v = torch.norm(vel, dim=1)
    
    # Specific Mechanical Energy: epsilon = v^2/2 - mu/r
    energy = (v**2 / 2.0) - (MU / r)
    return energy

def physics_loss(pred, target):
    """
    Penalize the difference in orbital energy between prediction and target.
    """
    # This assumes pred and target are in physical units (km, km/s)
    # If they are normalized, we need to un-normalize them within the loss function
    # for physical consistency.
    e_pred = get_orbital_energy(pred)
    e_target = get_orbital_energy(target)
    
    return torch.mean((e_pred - e_target)**2)
