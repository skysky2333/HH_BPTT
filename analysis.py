import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import random
from sklearn.model_selection import train_test_split

if torch.backends.mps.is_available():
    try:
        _ = torch.tensor([1.0, 2.0]).to("mps") * 2.0
        DEVICE = torch.device("mps")
    except Exception as e:
        print(f"MPS device found but test operation failed: {e}. Falling back to CPU.")
        DEVICE = torch.device("cpu")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


DEVICE = torch.device("cpu")
print(f"Using device for training: {DEVICE}")
DATA_GEN_DEVICE = torch.device("cpu")
print(f"Using device for data generation: {DATA_GEN_DEVICE}")


NEURON_PARAMS = {
    "squid_axon": {
        'g_Na': 120.0, 'g_K': 36.0, 'g_L': 0.3,
        'E_Na': 50.0, 'E_K': -77.0, 'E_L': -54.387,
        'C_m': 1.0, 'V0': -65.0
    },
    "cortical_RS": {
        'g_Na': 55.0, 'g_K': 15.0, 'g_L': 0.4,
        'E_Na': 50.0, 'E_K': -90.0, 'E_L': -70.0,
        'C_m': 1.0, 'V0': -65.0
    },
    "thalamic_TC": {
        'g_Na': 90.0, 'g_K': 10.0, 'g_L': 0.05,
        'E_Na': 50.0, 'E_K': -100.0, 'E_L': -70.0,
        'C_m': 1.0, 'V0': -65.0
    },
     "generic_fast_spiking": {
        'g_Na': 100.0, 'g_K': 50.0, 'g_L': 0.5,
        'E_Na': 50.0, 'E_K': -85.0, 'E_L': -65.0,
        'C_m': 0.9, 'V0': -65.0
    }
}

SELECTED_NEURON_TYPE = "squid_axon"
target_params_dict = NEURON_PARAMS[SELECTED_NEURON_TYPE]
print(f"Using parameters for: {SELECTED_NEURON_TYPE} (as ground truth)")

initialization_types = list(NEURON_PARAMS.keys())
initialization_types.remove(SELECTED_NEURON_TYPE)
print(f"Available types for initialization: {initialization_types}")

EPSILON = 1e-8

def alpha_m(V):
  V = torch.as_tensor(V, dtype=torch.float32)
  numerator = 0.1 * (V + 40.0)
  denominator = 1.0 - torch.exp(-(V + 40.0) / 10.0) + EPSILON
  return torch.where(torch.abs(V + 40.0) < 1e-6,
                     torch.tensor(1.0, dtype=torch.float32, device=V.device),
                     numerator / denominator)
def beta_m(V):
  V = torch.as_tensor(V, dtype=torch.float32)
  return 4.0 * torch.exp(-(V + 65.0) / 18.0)
def alpha_h(V):
  V = torch.as_tensor(V, dtype=torch.float32)
  return 0.07 * torch.exp(-(V + 65.0) / 20.0)
def beta_h(V):
  V = torch.as_tensor(V, dtype=torch.float32)
  return 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
def alpha_n(V):
  V = torch.as_tensor(V, dtype=torch.float32)
  numerator = 0.01 * (V + 55.0)
  denominator = 1.0 - torch.exp(-(V + 55.0) / 10.0) + EPSILON
  return torch.where(torch.abs(V + 55.0) < 1e-6,
                     torch.tensor(0.1, dtype=torch.float32, device=V.device),
                     numerator / denominator)
def beta_n(V):
  V = torch.as_tensor(V, dtype=torch.float32)
  return 0.125 * torch.exp(-(V + 65.0) / 80.0)


def simulate_hh_model(I_inj, t_span, dt, V0, params, device=DATA_GEN_DEVICE):
    """
    Simulates the HH model using Euler method on the specified device.
    Accepts a params dictionary for conductances and potentials.
    """
    g_Na = torch.tensor(params['g_Na'], dtype=torch.float32, device=device)
    g_K = torch.tensor(params['g_K'], dtype=torch.float32, device=device)
    g_L = torch.tensor(params['g_L'], dtype=torch.float32, device=device)
    E_Na = torch.tensor(params['E_Na'], dtype=torch.float32, device=device)
    E_K = torch.tensor(params['E_K'], dtype=torch.float32, device=device)
    E_L = torch.tensor(params['E_L'], dtype=torch.float32, device=device)
    C_m = torch.tensor(params['C_m'], dtype=torch.float32, device=device)
    V0_tensor = torch.tensor(V0, dtype=torch.float32, device=device)

    t = torch.arange(0, t_span + dt, dt, dtype=torch.float32, device=device)
    num_steps = len(t)

    if len(I_inj) != num_steps:
        if len(I_inj) < num_steps:
             padding = torch.full((num_steps - len(I_inj),), I_inj[-1].item(), dtype=torch.float32, device=device)
             I_inj = torch.cat((I_inj.to(device), padding))
        else:
             I_inj = I_inj[:num_steps].to(device)
    else:
        I_inj = I_inj.to(device)

    V = torch.full((num_steps,), V0_tensor.item(), dtype=torch.float32, device=device)
    m = torch.full((num_steps,), (alpha_m(V0_tensor) / (alpha_m(V0_tensor) + beta_m(V0_tensor))).item(), dtype=torch.float32, device=device)
    h = torch.full((num_steps,), (alpha_h(V0_tensor) / (alpha_h(V0_tensor) + beta_h(V0_tensor))).item(), dtype=torch.float32, device=device)
    n = torch.full((num_steps,), (alpha_n(V0_tensor) / (alpha_n(V0_tensor) + beta_n(V0_tensor))).item(), dtype=torch.float32, device=device)

    V_out = V.clone()
    m_out = m.clone()
    h_out = h.clone()
    n_out = n.clone()

    for i in range(num_steps - 1):
        V_i = V_out[i]; m_i = m_out[i]; h_i = h_out[i]; n_i = n_out[i]

        I_Na = g_Na * (m_i**3) * h_i * (V_i - E_Na)
        I_K = g_K * (n_i**4) * (V_i - E_K)
        I_L = g_L * (V_i - E_L)
        I_ion = I_Na + I_K + I_L

        am, bm = alpha_m(V_i), beta_m(V_i)
        ah, bh = alpha_h(V_i), beta_h(V_i)
        an, bn = alpha_n(V_i), beta_n(V_i)

        m_next = m_i + dt * (am * (1.0 - m_i) - bm * m_i)
        h_next = h_i + dt * (ah * (1.0 - h_i) - bh * h_i)
        n_next = n_i + dt * (an * (1.0 - n_i) - bn * n_i)
        V_next = V_i + dt * (I_inj[i] - I_ion) / C_m

        m_out[i+1] = m_next
        h_out[i+1] = h_next
        n_out[i+1] = n_next
        V_out[i+1] = V_next

    return t, V_out, m_out, h_out, n_out


def generate_data(n_samples, t_span, dt, current_range, base_params,
                  generation_device, voltage_noise_std=0.0, param_noise_std=0.0):
    """
    Generates HH simulation data.
    Adds optional Gaussian noise to the voltage trace.
    Adds optional Gaussian noise to conductance parameters for each simulation.
    Stores results on CPU.
    """
    all_data = []
    input_currents = torch.linspace(current_range[0], current_range[1], n_samples, device=generation_device)
    num_time_steps = int(t_span / dt) + 1

    print(f"Generating {n_samples} samples on {generation_device} (V_noise={voltage_noise_std}, P_noise={param_noise_std})...")
    for i in range(n_samples):
        current_level = input_currents[i]
        I_inj_vec = torch.full((num_time_steps,), current_level.item(), dtype=torch.float32, device=generation_device)

        sim_params = copy.deepcopy(base_params)
        if param_noise_std > 0:
            noise_gNa = 1.0 + torch.randn(1).item() * param_noise_std
            noise_gK  = 1.0 + torch.randn(1).item() * param_noise_std
            noise_gL  = 1.0 + torch.randn(1).item() * param_noise_std
            sim_params['g_Na'] = max(0.0, sim_params['g_Na'] * noise_gNa)
            sim_params['g_K']  = max(0.0, sim_params['g_K']  * noise_gK)
            sim_params['g_L']  = max(0.0, sim_params['g_L']  * noise_gL)

        t, V, _, _, _ = simulate_hh_model(I_inj_vec, t_span, dt, base_params['V0'], sim_params, device=generation_device)

        if voltage_noise_std > 0:
            noise = torch.randn_like(V) * voltage_noise_std
            V = V + noise

        all_data.append({'input_current': I_inj_vec.cpu(), 'voltage_trace': V.cpu()})
        if (i + 1) % max(1, n_samples // 5) == 0 or i == n_samples - 1:
             print(f"  Generated sample {i+1}/{n_samples}")

    return all_data

class UnrolledHH(nn.Module):
    """ Unrolled HH simulation with learnable conductances. Optimized forward pass. """
    def __init__(self, dt, V0, E_Na, E_K, E_L, C_m, initial_params=None, device=DEVICE):
        """
        Initializes the model.

        Args:
            dt (float): Simulation time step.
            V0 (float): Initial membrane potential.
            E_Na (float): Sodium reversal potential.
            E_K (float): Potassium reversal potential.
            E_L (float): Leak reversal potential.
            C_m (float): Membrane capacitance.
            initial_params (dict, optional): Dictionary specifying {'g_Na', 'g_K', 'g_L'}
                                             for initialization. If None, initializes
                                             near target defaults with noise.
            device (torch.device): Device to run the model on.
        """
        super().__init__()
        self.dt = torch.tensor(dt, dtype=torch.float32, device=device)
        self.V0 = torch.tensor(V0, dtype=torch.float32, device=device)
        self.E_Na = torch.tensor(E_Na, dtype=torch.float32, device=device)
        self.E_K = torch.tensor(E_K, dtype=torch.float32, device=device)
        self.E_L = torch.tensor(E_L, dtype=torch.float32, device=device)
        self.C_m = torch.tensor(C_m, dtype=torch.float32, device=device)
        self.device = device

        if initial_params is not None:
            print(f"Initializing parameters from provided dict: {initial_params}")
            init_g_Na = torch.tensor(initial_params['g_Na'], dtype=torch.float32)
            init_g_K  = torch.tensor(initial_params['g_K'], dtype=torch.float32)
            init_g_L  = torch.tensor(initial_params['g_L'], dtype=torch.float32)
        else:
            print("Initializing parameters near target values with noise (Default).")
            target_g_Na = NEURON_PARAMS[SELECTED_NEURON_TYPE]['g_Na']
            target_g_K  = NEURON_PARAMS[SELECTED_NEURON_TYPE]['g_K']
            target_g_L  = NEURON_PARAMS[SELECTED_NEURON_TYPE]['g_L']
            init_g_Na = target_g_Na * (1.0 + 0.1 * torch.randn(1))
            init_g_K  = target_g_K  * (1.0 + 0.1 * torch.randn(1))
            init_g_L  = target_g_L  * (1.0 + 0.1 * torch.randn(1))

        self.raw_g_Na = nn.Parameter(init_g_Na.clone().detach().to(self.device))
        self.raw_g_K  = nn.Parameter(init_g_K.clone().detach().to(self.device))
        self.raw_g_L  = nn.Parameter(init_g_L.clone().detach().to(self.device))


    def forward(self, I_inj):
        """ Performs the forward pass using list accumulation and torch.stack. """
        I_inj = I_inj.to(self.device)
        is_batched = I_inj.dim() == 2
        if not is_batched:
            I_inj = I_inj.unsqueeze(0)

        batch_size, num_steps = I_inj.shape

        g_Na = nn.functional.softplus(self.raw_g_Na)
        g_K = nn.functional.softplus(self.raw_g_K)
        g_L = nn.functional.softplus(self.raw_g_L)

        g_Na = g_Na.expand(batch_size); g_K = g_K.expand(batch_size); g_L = g_L.expand(batch_size)
        E_Na = self.E_Na.expand(batch_size); E_K = self.E_K.expand(batch_size); E_L = self.E_L.expand(batch_size)
        C_m = self.C_m.expand(batch_size)

        V_current = torch.full((batch_size,), self.V0.item(), dtype=torch.float32, device=self.device)
        m0 = alpha_m(self.V0) / (alpha_m(self.V0) + beta_m(self.V0))
        h0 = alpha_h(self.V0) / (alpha_h(self.V0) + beta_h(self.V0))
        n0 = alpha_n(self.V0) / (alpha_n(self.V0) + beta_n(self.V0))
        m_current = torch.full_like(V_current, m0.item())
        h_current = torch.full_like(V_current, h0.item())
        n_current = torch.full_like(V_current, n0.item())

        V_history = [V_current]

        for i in range(num_steps - 1):
            V_i = V_current; m_i = m_current; h_i = h_current; n_i = n_current
            I_inj_i = I_inj[:, i]

            am, bm = alpha_m(V_i), beta_m(V_i); ah, bh = alpha_h(V_i), beta_h(V_i); an, bn = alpha_n(V_i), beta_n(V_i)
            I_Na = g_Na * (m_i**3) * h_i * (V_i - E_Na); I_K = g_K * (n_i**4) * (V_i - E_K); I_L = g_L * (V_i - E_L)
            I_ion = I_Na + I_K + I_L

            m_next = m_i + self.dt * (am * (1.0 - m_i) - bm * m_i)
            h_next = h_i + self.dt * (ah * (1.0 - h_i) - bh * h_i)
            n_next = n_i + self.dt * (an * (1.0 - n_i) - bn * n_i)
            V_next = V_i + self.dt * (I_inj_i - I_ion) / C_m

            if not torch.isfinite(V_next).all():
                print(f"Warning: Non-finite voltage ({V_next.detach().cpu().numpy()}) detected at step {i+1}. Stopping simulation early for this batch.")
                last_V = V_history[-1]
                remaining_steps = num_steps - len(V_history)
                padding_list = [last_V] * remaining_steps
                V_history.extend(padding_list)
                break

            V_current = V_next; m_current = m_next; h_current = h_next; n_current = n_next
            V_history.append(V_current)

        if len(V_history) < num_steps:
             last_V = V_history[-1]
             remaining_steps = num_steps - len(V_history)
             padding_list = [last_V] * remaining_steps
             V_history.extend(padding_list)

        if len(V_history) > 0:
            expected_size = V_history[0].size()
            V_history_squeezed = [t.squeeze() if t.ndim > 1 else t for t in V_history]
            try:
                V = torch.stack(V_history_squeezed, dim=1)
            except RuntimeError as e:
                 print(f"Error stacking V_history: {e}. History length: {len(V_history)}. Sizes: {[t.size() for t in V_history_squeezed]}")
                 V = torch.zeros((batch_size, num_steps), device=self.device)
        else:
             V = torch.empty((batch_size, 0), device=self.device)

        if not is_batched:
            V = V.squeeze(0)

        return V

    def get_params(self):
        """ Returns the estimated conductances (scalar Python floats). """
        return {
            'g_Na': nn.functional.softplus(self.raw_g_Na).item(),
            'g_K': nn.functional.softplus(self.raw_g_K).item(),
            'g_L': nn.functional.softplus(self.raw_g_L).item()
        }


def train_model(model, train_data, test_data, optimizer, scheduler, criterion, epochs, device, clip_value=1.0, print_every_samples=10):
    """ Trains the UnrolledHH model with LR scheduler, gradient clipping, and parameter tracking. """
    train_losses = []
    test_losses = []
    param_history = {'g_Na': [], 'g_K': [], 'g_L': []}
    start_time = time.time()

    n_train_samples = len(train_data)
    n_test_samples = len(test_data)

    print(f"Starting Training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        finite_train_count = 0
        processed_samples = 0

        for i, sample in enumerate(train_data):
            I_inj_true = sample['input_current'].to(device)
            V_true = sample['voltage_trace'].to(device)

            optimizer.zero_grad()
            V_pred = model(I_inj_true)

            if not torch.isfinite(V_pred).all():
                 print(f"Warning: NaN/Inf prediction detected at epoch {epoch+1}, train sample {i}. Skipping update.")
                 continue

            loss = criterion(V_pred, V_true)

            if not torch.isfinite(loss):
                print(f"Warning: NaN/Inf loss ({loss.item()}) detected at epoch {epoch+1}, train sample {i}. Skipping update.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            optimizer.step()

            epoch_train_loss += loss.item()
            finite_train_count += 1
            processed_samples += 1

            if (processed_samples % print_every_samples == 0) and processed_samples > 0:
                 current_lr = optimizer.param_groups[0]['lr']
                 print(f"  Epoch {epoch+1}/{epochs} | Sample {processed_samples}/{n_train_samples} | Batch Loss: {loss.item():.6f} | LR: {current_lr:.6f}")


        avg_train_loss = epoch_train_loss / finite_train_count if finite_train_count > 0 else float('nan')
        train_losses.append(avg_train_loss)

        current_params = model.get_params()
        param_history['g_Na'].append(current_params['g_Na'])
        param_history['g_K'].append(current_params['g_K'])
        param_history['g_L'].append(current_params['g_L'])

        model.eval()
        epoch_test_loss = 0.0
        finite_test_count = 0
        with torch.no_grad():
            for sample in test_data:
                I_inj_true = sample['input_current'].to(device)
                V_true = sample['voltage_trace'].to(device)
                V_pred = model(I_inj_true)
                if not torch.isfinite(V_pred).all():
                    print(f"Warning: NaN/Inf prediction during validation (epoch {epoch+1}). Skipping sample.")
                    continue
                loss = criterion(V_pred, V_true)
                if torch.isfinite(loss):
                    epoch_test_loss += loss.item()
                    finite_test_count += 1
                else:
                     print(f"Warning: NaN/Inf loss during validation (epoch {epoch+1}). Skipping sample.")


        avg_test_loss = epoch_test_loss / finite_test_count if finite_test_count > 0 else float('nan')
        test_losses.append(avg_test_loss)

        if scheduler is not None:
             if np.isfinite(avg_test_loss):
                 scheduler.step(avg_test_loss)
             else:
                 print(f"Warning: Invalid validation loss ({avg_test_loss}) at epoch {epoch+1}. Scheduler not stepped.")


        if (epoch + 1) % 25 == 0 or epoch == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] Summary: "
                f"Avg Train Loss: {avg_train_loss:.6f}, "
                f"Avg Test Loss: {avg_test_loss:.6f}, "
                f"LR: {current_lr:.6f}")
            print(f"  Params (g_Na, g_K, g_L): ({current_params['g_Na']:.3f}, {current_params['g_K']:.3f}, {current_params['g_L']:.3f})")

        if not all(np.isfinite(p) for p in current_params.values()):
            print(f"Stopping early due to non-finite parameters at epoch {epoch+1}.")
            remaining_epochs = epochs - (epoch + 1)
            for key in param_history: param_history[key].extend([float('nan')] * remaining_epochs)
            train_losses.extend([float('nan')] * remaining_epochs); test_losses.extend([float('nan')] * remaining_epochs)
            break

    end_time = time.time()
    print(f"\nTraining Finished in {end_time - start_time:.2f} seconds.")
    final_epochs = len(train_losses)
    if final_epochs < epochs:
        last_train_loss = train_losses[-1] if train_losses and np.isfinite(train_losses[-1]) else float('nan')
        last_test_loss = test_losses[-1] if test_losses and np.isfinite(test_losses[-1]) else float('nan')
        train_losses.extend([last_train_loss] * (epochs - final_epochs))
        test_losses.extend([last_test_loss] * (epochs - final_epochs))
        for key in param_history:
            last_param = param_history[key][-1] if param_history[key] and np.isfinite(param_history[key][-1]) else float('nan')
            param_history[key].extend([last_param] * (epochs - final_epochs))

    return train_losses, test_losses, param_history


def plot_loss_curve(epochs, train_losses, test_losses):
    """ Plots the training and testing loss curves. """
    plt.figure(figsize=(10, 5))
    valid_epochs = range(1, epochs + 1)
    finite_losses = [l for l in train_losses + test_losses if l is not None and np.isfinite(l) and l > 0]
    min_loss = min(finite_losses) / 10 if finite_losses else 1e-6

    plt.plot(valid_epochs, train_losses, label='Training Loss', marker='.', linestyle='-')
    plt.plot(valid_epochs, test_losses, label='Test Loss', marker='.', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.ylim(bottom=min_loss)
    plt.show()

def plot_parameter_trajectory(epochs, param_history, target_params):
    """ Plots the evolution of estimated parameters over epochs. """
    plt.figure(figsize=(12, 6))
    epochs_range = range(1, epochs + 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_history)))

    for i, (param_name, history) in enumerate(param_history.items()):
        valid_epochs_param = [e for e, h in zip(epochs_range, history) if h is not None and np.isfinite(h)]
        valid_history = [h for h in history if h is not None and np.isfinite(h)]

        if valid_history:
            plt.plot(valid_epochs_param, valid_history, label=f'Estimated {param_name}', marker='.', linestyle='-', color=colors[i])
        plt.axhline(y=target_params[param_name], linestyle='--', color=colors[i], alpha=0.7, label=f'Target {param_name}' if i == 0 else "_nolegend_")


    plt.xlabel('Epoch')
    plt.ylabel('Conductance (mS/cm^2)')
    plt.title('Parameter Estimation Trajectory During Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_comparison(model, data_sample, t_span, dt, eval_device, title="Test Sample Comparison"):
    """ Plots the ground truth vs predicted voltage for a single sample. """
    model.eval()
    I_inj_true = data_sample['input_current'].to(eval_device)
    V_true_device = data_sample['voltage_trace'].to(eval_device)

    with torch.no_grad():
        V_pred_device = model(I_inj_true)

    V_true_cpu = data_sample['voltage_trace'].cpu()
    V_pred_cpu = V_pred_device.cpu()
    I_inj_cpu = data_sample['input_current'].cpu()

    time_vec = torch.arange(0, t_span + dt, dt)

    plt.figure(figsize=(12, 6))
    plt.plot(time_vec.numpy(), V_true_cpu.numpy(), label='Ground Truth Voltage', color='blue', linewidth=2, alpha=0.8)
    plt.plot(time_vec.numpy(), V_pred_cpu.numpy(), label='Predicted Voltage', color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.grid(True)

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    color = 'gray'
    ax2.set_ylabel('Input Current (uA/cm^2)', color=color)
    ax2.plot(time_vec.numpy()[:len(I_inj_cpu)], I_inj_cpu.numpy(), color=color, linestyle=':', label='Input Current')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=min(0, I_inj_cpu.min().item()-1))
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def calculate_relative_errors(estimated_params, target_params):
    """ Calculates the relative error for each parameter as a percentage. """
    errors = {}
    for name, target_val in target_params.items():
        if name not in ['g_Na', 'g_K', 'g_L']: continue

        est_val = estimated_params.get(name)
        if est_val is None or not np.isfinite(est_val):
             errors[name] = float('nan')
             continue

        if target_val != 0:
            errors[name] = abs(est_val - target_val) / abs(target_val) * 100
        elif target_val == 0 and est_val == 0:
            errors[name] = 0.0
        else:
            errors[name] = float('inf')
    return errors


def run_noise_analysis(noise_config, data_config, train_config, initialization_types):
    """
    Runs the full analysis across different noise levels.
    Generates data based on SELECTED_NEURON_TYPE.
    Initializes model parameters based on a *different*, randomly chosen neuron type.
    """
    results_list = []
    target_params = NEURON_PARAMS[SELECTED_NEURON_TYPE]
    training_device = train_config['DEVICE']

    for v_noise, p_noise in noise_config['LEVELS']:
        print(f"\n--- Running Noise Analysis for V_Noise={v_noise:.2f}, P_Noise={p_noise:.2f} ---")

        n_total_samples = data_config['N_SAMPLES_TRAIN'] + data_config['N_SAMPLES_TEST']
        all_generated_data = generate_data(
            n_samples=n_total_samples,
            t_span=data_config['T_SPAN'],
            dt=data_config['DT'],
            current_range=data_config['CURRENT_RANGE_ALL'],
            base_params=target_params,
            generation_device=data_config['DATA_GEN_DEVICE'],
            voltage_noise_std=v_noise,
            param_noise_std=p_noise
        )

        if not all_generated_data or len(all_generated_data) < n_total_samples:
            print(f"Error: Insufficient data generated. Skipping.")
            continue
        if data_config['N_SAMPLES_TEST'] >= len(all_generated_data):
             print(f"Error: Test size >= total samples. Skipping.")
             continue
        train_data, test_data = train_test_split(
            all_generated_data, test_size=data_config['N_SAMPLES_TEST'], random_state=42
        )
        print(f"Split data: {len(train_data)} train, {len(test_data)} test samples.")

        if not initialization_types:
             print("Warning: No alternative neuron types available for initialization. Using default near-target init.")
             init_params_dict = None
        else:
            init_type = random.choice(initialization_types)
            print(f"Initializing model with parameters from: {init_type}")
            init_params_dict = NEURON_PARAMS[init_type]

        model = UnrolledHH(
            dt=data_config['DT'],
            V0=target_params['V0'],
            E_Na=target_params['E_Na'], E_K=target_params['E_K'],
            E_L=target_params['E_L'], C_m=target_params['C_m'],
            device=training_device,
            initial_params=init_params_dict
        )
        model.to(training_device)
        optimizer = optim.Adam(model.parameters(), lr=train_config['LEARNING_RATE'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=train_config['SCHEDULER_FACTOR'], patience=train_config['SCHEDULER_PATIENCE'], verbose=True)
        criterion = nn.MSELoss()

        print("\nInitial Parameter Estimates (g_Na, g_K, g_L):")
        print(f"{model.get_params()}")
        print(f"Target Parameters: {{'g_Na': {target_params['g_Na']:.1f}, 'g_K': {target_params['g_K']:.1f}, 'g_L': {target_params['g_L']:.1f}}}")


        train_losses, test_losses, param_history = train_model(
            model, train_data, test_data, optimizer, scheduler, criterion,
            train_config['EPOCHS'], training_device, train_config['CLIP_VALUE'],
            train_config['PRINT_EVERY_SAMPLES']
        )

        final_params = model.get_params()
        rel_errors = calculate_relative_errors(final_params, target_params)

        results_list.append({
            'voltage_noise_std': v_noise,
            'param_noise_std': p_noise,
            'final_params': final_params,
            'rel_errors': rel_errors,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'param_history': param_history,
            'example_test_input': test_data[0]['input_current'].cpu() if test_data else None,
            'example_test_true_V': test_data[0]['voltage_trace'].cpu() if test_data else None,
            'example_test_pred_V': model(test_data[0]['input_current'].to(training_device)).detach().cpu() if test_data else None
        })

        print(f"\nResults for V_Noise={v_noise:.2f}, P_Noise={p_noise:.2f}:")
        print(f"  Final Params: {final_params}")
        print(f"  Relative Errors (%): {rel_errors}")

        print(f"\nPlotting results for V_Noise={v_noise:.2f}, P_Noise={p_noise:.2f}...")
        plot_loss_curve(train_config['EPOCHS'], train_losses, test_losses)
        plot_parameter_trajectory(train_config['EPOCHS'], param_history, target_params)
        if test_data:
            plot_comparison(model, test_data[0], data_config['T_SPAN'], data_config['DT'], training_device,
                            title=f"Test Sample 0 (V_Noise={v_noise:.2f}, P_Noise={p_noise:.2f})")

    return results_list

def run_datasize_analysis(datasize_config, data_config, train_config, initialization_types):
    """
    Runs analysis varying training dataset size at a fixed noise level.
    Uses initialization from a randomly chosen *different* neuron type.
    """
    results_list = []
    target_params = NEURON_PARAMS[SELECTED_NEURON_TYPE]
    training_device = train_config['DEVICE']
    v_noise, p_noise = datasize_config['FIXED_NOISE_LEVEL']
    n_test_samples = data_config['N_SAMPLES_TEST']

    print(f"\n--- Running Data Size Analysis (Fixed Noise: V={v_noise:.2f}, P={p_noise:.2f}) ---")

    max_train_size = max(datasize_config['TRAIN_SIZES'])
    n_pool_samples = max_train_size + n_test_samples + 50
    print(f"Generating data pool ({n_pool_samples} samples) for data size analysis...")
    all_generated_data = generate_data(
        n_samples=n_pool_samples,
        t_span=data_config['T_SPAN'], dt=data_config['DT'],
        current_range=data_config['CURRENT_RANGE_ALL'],
        base_params=target_params, generation_device=data_config['DATA_GEN_DEVICE'],
        voltage_noise_std=v_noise, param_noise_std=p_noise
    )

    if len(all_generated_data) < n_test_samples:
         print("Error: Not enough data generated for test set. Aborting data size analysis.")
         return []

    data_pool, test_data = train_test_split(all_generated_data, test_size=n_test_samples, random_state=42)
    print(f"Created fixed test set ({len(test_data)} samples) and training pool ({len(data_pool)} samples).")

    for train_size in datasize_config['TRAIN_SIZES']:
        print(f"\n--- Training with Data Size: {train_size} ---")
        if train_size > len(data_pool):
            print(f"Warning: Requested train size ({train_size}) > available pool ({len(data_pool)}). Skipping.")
            continue

        train_data = random.sample(data_pool, train_size)
        print(f"Using {len(train_data)} training samples.")

        if not initialization_types:
             print("Warning: No alternative neuron types available for initialization. Using default near-target init.")
             init_params_dict = None
        else:
            init_type = random.choice(initialization_types)
            print(f"Initializing model with parameters from: {init_type}")
            init_params_dict = NEURON_PARAMS[init_type]

        model = UnrolledHH(
            dt=data_config['DT'],
            V0=target_params['V0'], E_Na=target_params['E_Na'], E_K=target_params['E_K'],
            E_L=target_params['E_L'], C_m=target_params['C_m'],
            device=training_device,
            initial_params=init_params_dict
        )
        model.to(training_device)
        optimizer = optim.Adam(model.parameters(), lr=train_config['LEARNING_RATE'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=train_config['SCHEDULER_FACTOR'], patience=train_config['SCHEDULER_PATIENCE'], verbose=True)
        criterion = nn.MSELoss()

        print(f"Target Parameters: {{'g_Na': {target_params['g_Na']:.1f}, 'g_K': {target_params['g_K']:.1f}, 'g_L': {target_params['g_L']:.1f}}}")

        train_losses, test_losses, param_history = train_model(
            model, train_data, test_data, optimizer, scheduler, criterion,
            train_config['EPOCHS'], training_device, train_config['CLIP_VALUE'],
            max(1, train_size // 5)
        )

        final_params = model.get_params()
        rel_errors = calculate_relative_errors(final_params, target_params)

        results_list.append({
            'train_size': train_size,
            'final_params': final_params,
            'rel_errors': rel_errors,
        })

        print(f"\nResults for Train Size={train_size}:")
        print(f"  Final Params: {final_params}")
        print(f"  Relative Errors (%): {rel_errors}")

    return results_list


def plot_noise_robustness(results_list):
    """ Plots parameter relative errors vs noise level (assuming one noise type varies). """
    noise_levels = sorted(list(set(res['voltage_noise_std'] for res in results_list)))
    param_names = ['g_Na', 'g_K', 'g_L']

    plt.figure(figsize=(10, 6))

    for param_name in param_names:
        errors_vs_noise = []
        valid_noise_levels = []
        for v_noise in noise_levels:
             errors_at_level = [res['rel_errors'].get(param_name, float('nan'))
                                for res in results_list if res['voltage_noise_std'] == v_noise and res['param_noise_std'] == 0.0]
             finite_errors = [e for e in errors_at_level if np.isfinite(e)]
             if finite_errors:
                 errors_vs_noise.append(np.mean(finite_errors))
                 valid_noise_levels.append(v_noise)

        if valid_noise_levels:
            plt.plot(valid_noise_levels, errors_vs_noise, marker='o', linestyle='-', label=f'{param_name} Rel. Error')
        else:
            print(f"No finite error data to plot for parameter {param_name} vs voltage noise.")

    plt.xlabel('Voltage Noise Standard Deviation (mV)')
    plt.ylabel('Relative Parameter Error (%)')
    plt.title('Parameter Estimation Error vs. Voltage Noise Level (Param Noise = 0)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()

    param_noise_levels = sorted(list(set(res['param_noise_std'] for res in results_list)))
    plt.figure(figsize=(10, 6))

    for param_name in param_names:
        errors_vs_pnoise = []
        valid_pnoise_levels = []
        for p_noise in param_noise_levels:
             errors_at_level = [res['rel_errors'].get(param_name, float('nan'))
                                for res in results_list if res['param_noise_std'] == p_noise and res['voltage_noise_std'] == 0.0]
             finite_errors = [e for e in errors_at_level if np.isfinite(e)]
             if finite_errors:
                 errors_vs_pnoise.append(np.mean(finite_errors))
                 valid_pnoise_levels.append(p_noise)

        if valid_pnoise_levels:
            plt.plot(valid_pnoise_levels, errors_vs_pnoise, marker='s', linestyle='--', label=f'{param_name} Rel. Error')
        else:
            print(f"No finite error data to plot for parameter {param_name} vs parameter noise.")

    plt.xlabel('Parameter Noise Standard Deviation (Relative)')
    plt.ylabel('Relative Parameter Error (%)')
    plt.title('Parameter Estimation Error vs. Parameter Noise Level (Voltage Noise = 0)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()

def plot_datasize_performance(datasize_results, fixed_noise_level):
    """ Plots parameter relative errors vs training dataset size. """
    plt.figure(figsize=(10, 6))
    train_sizes = [res['train_size'] for res in datasize_results]
    param_names = ['g_Na', 'g_K', 'g_L']

    for param_name in param_names:
        errors = [res['rel_errors'].get(param_name, float('nan')) for res in datasize_results]
        valid_sizes = [s for s, e in zip(train_sizes, errors) if np.isfinite(e)]
        valid_errors = [e for e in errors if np.isfinite(e)]
        if valid_errors:
            plt.plot(valid_sizes, valid_errors, marker='o', linestyle='-', label=f'{param_name} Rel. Error')
        else:
            print(f"No finite error data to plot for parameter {param_name} vs data size.")

    plt.xlabel('Number of Training Samples')
    plt.ylabel('Relative Parameter Error (%)')
    plt.title(f'Parameter Estimation Error vs. Training Data Size (Noise: V={fixed_noise_level[0]:.1f}, P={fixed_noise_level[1]:.2f})')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xscale('log')
    valid_train_sizes_in_results = sorted(list(set(valid_sizes)))
    plt.xticks(valid_train_sizes_in_results, labels=[str(s) for s in valid_train_sizes_in_results])
    plt.minorticks_off()
    plt.show()


data_config = {
    'T_SPAN': 50.0,
    'DT': 0.05,
    'N_SAMPLES_TRAIN': 10,
    'N_SAMPLES_TEST': 3,
    'CURRENT_RANGE_ALL': (5.0, 15.0),
    'DATA_GEN_DEVICE': DATA_GEN_DEVICE,
}

train_config = {
    'LEARNING_RATE': 0.5,
    'EPOCHS': 300,
    'CLIP_VALUE': 0.5,
    'DEVICE': DEVICE,
    'PRINT_EVERY_SAMPLES': 100,
    'SCHEDULER_PATIENCE': 50,
    'SCHEDULER_FACTOR': 0.5,
    'USE_RANDOM_INIT': True
}


noise_config = {
    'LEVELS': [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
        (0.0, 0.05),
        (0.0, 0.1),
        (0.0, 0.2),
        (0.0, 0.4),
    ]
}

datasize_config = {
    'TRAIN_SIZES': [1, 2, 5, 10, 25, 50, 100],
    'FIXED_NOISE_LEVEL': (1.0, 0.1)
}


print(f"--- Running Noise Analysis for Neuron Type: {SELECTED_NEURON_TYPE} ---")
init_method_noise = initialization_types if train_config['USE_RANDOM_INIT'] else None
noise_analysis_results = run_noise_analysis(noise_config, data_config, train_config, init_method_noise)

print("\n--- Plotting Overall Noise Robustness ---")
plot_noise_robustness(noise_analysis_results)

print(f"\n--- Running Data Size Analysis for Neuron Type: {SELECTED_NEURON_TYPE} ---")
init_method_datasize = initialization_types if train_config['USE_RANDOM_INIT'] else None
datasize_analysis_results = run_datasize_analysis(datasize_config, data_config, train_config, init_method_datasize)

print("\n--- Plotting Data Size Performance ---")
plot_datasize_performance(datasize_analysis_results, datasize_config['FIXED_NOISE_LEVEL'])


print("\n--- Comprehensive Analysis Complete ---")

