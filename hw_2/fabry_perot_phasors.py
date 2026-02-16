import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import patches

# Fabry-Perot parameters
r = 0.9  # Mirror reflectivity (amplitude)
t = np.sqrt(1 - r**2)  # Transmission coefficient
n_reflections = 8  # Number of reflections to show

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def draw_phasor(ax, start, end, color, label=None, width=2, alpha=1.0):
    """Draw a phasor arrow from start to end"""
    arrow = FancyArrowPatch(start, end,
                           arrowstyle='->', 
                           mutation_scale=20,
                           linewidth=width,
                           color=color,
                           alpha=alpha,
                           label=label)
    ax.add_patch(arrow)

def plot_phasors(ax, phi_rt, title):
    """Plot phasor diagram for given round-trip phase"""
    
    # Calculate individual field phasors
    E_fields = []
    amplitudes = []
    phases = []
    
    # Starting point
    origin = np.array([0, 0])
    current_pos = origin.copy()
    
    for n in range(n_reflections):
        # Amplitude decreases with each round trip
        amplitude = t * r**n
        amplitudes.append(amplitude)
        
        # Phase advances by phi_rt each round trip
        phase = n * phi_rt
        phases.append(phase)
        
        # Field vector
        E_x = amplitude * np.cos(phase)
        E_y = amplitude * np.sin(phase)
        E_fields.append([E_x, E_y])
        
        # Draw individual phasor
        next_pos = current_pos + np.array([E_x, E_y])
        
        if n < 5:  # Only label first few
            color = plt.cm.viridis(n / 8)
            draw_phasor(ax, current_pos, next_pos, color, 
                       label=f'$E_{n}$', width=1.5, alpha=0.8)
        else:
            color = plt.cm.viridis(n / 8)
            draw_phasor(ax, current_pos, next_pos, color, width=1.5, alpha=0.6)
        
        current_pos = next_pos
    
    # Total field (cavity field)
    E_total = current_pos
    
    # Draw total field phasor from origin
    draw_phasor(ax, origin, E_total, 'red', 
               label='$E_{cav}$ (total)', width=3, alpha=1.0)
    
    # Draw dashed line from origin to show total
    ax.plot([0, E_total[0]], [0, E_total[1]], 'r--', linewidth=1, alpha=0.3)
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Re($E$)', fontsize=12)
    ax.set_ylabel('Im($E$)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    # Only show unique labels
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    
    # Set limits
    max_val = max(1.5, abs(E_total[0]) + 0.5, abs(E_total[1]) + 0.5)
    ax.set_xlim([-0.5, max_val])
    ax.set_ylim([-max_val*0.3, max_val])
    
    # Add text showing total field magnitude
    E_mag = np.sqrt(E_total[0]**2 + E_total[1]**2)
    ax.text(0.05, 0.95, f'$|E_{{cav}}| = {E_mag:.3f}$', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Case 1: On resonance (phi_rt = 0)
plot_phasors(ax1, phi_rt=0, title='On Resonance: $\\phi_{rt} = 0$')

# Case 2: Slightly off resonance (phi_rt = 0.3 rad)
phi_off = 0.3  # Small phase, phi_rt << 1
plot_phasors(ax2, phi_rt=phi_off, title=f'Off Resonance: $\\phi_{{rt}} = {phi_off}$ rad')

plt.tight_layout()
plt.show()

