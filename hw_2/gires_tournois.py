import numpy as np
import matplotlib.pyplot as plt

# Physical constants
c = 3e8  # speed of light (m/s)

# Cavity parameters
L = 0.01  # cavity length (1 cm)
FSR = c / (2 * L)  # Free spectral range in Hz
print(f"Cavity length: {L*100} cm")
print(f"FSR: {FSR/1e9:.2f} GHz")

# Frequency array centered around a resonance
nu_0 = 10 * FSR  # Center frequency (arbitrary)
delta_nu = np.linspace(-2*FSR, 2*FSR, 2000)  # Scan ±2 FSR
nu = nu_0 + delta_nu

# Round-trip phase
phi = 4 * np.pi * nu * L / c  # 2kL = 4πνL/c

# Different R1 values to plot
R1_values = [0.3, 0.5, 0.7, 0.9, 0.95]

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

colors = plt.cm.viridis(np.linspace(0, 1, len(R1_values)))

for i, R1 in enumerate(R1_values):
    r1 = np.sqrt(R1)  # amplitude reflectivity
    r2 = 1.0  # R2 = 100%
    
    # Complex reflectivity: (-r1 + e^(iφ)) / (1 - r1*e^(iφ))
    # Numerator: -r1 + e^(iφ)
    numerator = -r1 + np.exp(1j * phi)
    
    # Denominator: 1 - r1*e^(iφ)
    denominator = 1 - r1 * np.exp(1j * phi)
    
    # Complex reflectivity
    rho = numerator / denominator
    
    # Magnitude and phase
    magnitude = np.abs(rho)
    phase = np.angle(rho)  # in radians
    
    # Unwrap phase to avoid discontinuities
    phase_unwrapped = np.unwrap(phase)
    
    # Convert to degrees for plotting
    phase_deg = np.degrees(phase)
    phase_unwrapped_deg = np.degrees(phase_unwrapped)
    
    # Plot 1: Magnitude (should be 1)
    ax1.plot(delta_nu / FSR, magnitude, color=colors[i], 
             label=f'$R_1 = {R1}$', linewidth=2)
    
    # Plot 2: Phase (wrapped)
    ax2.plot(delta_nu / FSR, phase_deg, color=colors[i], 
             label=f'$R_1 = {R1}$', linewidth=2)
    
    # Plot 3: Phase (unwrapped)
    ax3.plot(delta_nu / FSR, phase_unwrapped_deg, color=colors[i], 
             label=f'$R_1 = {R1}$', linewidth=2)

# Format plot 1: Magnitude
ax1.set_ylabel('$|\\tilde{E}_{refl}/\\tilde{E}_{inc}|$', fontsize=14)
ax1.set_xlabel('Frequency offset (FSR)', fontsize=12)
ax1.set_title('Gires-Ournois Interferometer: Reflection Magnitude vs Frequency', 
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10)
ax1.set_ylim([0.98, 1.02])
ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)

# Add secondary x-axis for absolute frequency
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xlabel('Frequency offset (GHz)', fontsize=12, color='blue')
ax1_top.tick_params(axis='x', labelcolor='blue')
# Convert FSR units to GHz
x_ticks_fsr = ax1.get_xticks()
x_ticks_ghz = x_ticks_fsr * FSR / 1e9
ax1_top.set_xticks(x_ticks_fsr)
ax1_top.set_xticklabels([f'{x:.0f}' for x in x_ticks_ghz])

# Format plot 2: Phase (wrapped)
ax2.set_ylabel('Phase (degrees)', fontsize=14)
ax2.set_xlabel('Frequency offset (FSR)', fontsize=12)
ax2.set_title('Reflection Phase vs Frequency (Wrapped)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=10)
ax2.set_ylim([-180, 180])
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Add secondary x-axis for absolute frequency
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
ax2_top.set_xlabel('Frequency offset (GHz)', fontsize=12, color='blue')
ax2_top.tick_params(axis='x', labelcolor='blue')
x_ticks_fsr = ax2.get_xticks()
x_ticks_ghz = x_ticks_fsr * FSR / 1e9
ax2_top.set_xticks(x_ticks_fsr)
ax2_top.set_xticklabels([f'{x:.0f}' for x in x_ticks_ghz])

# Format plot 3: Phase (unwrapped)
ax3.set_ylabel('Phase (degrees)', fontsize=14)
ax3.set_xlabel('Frequency offset (FSR)', fontsize=12)
ax3.set_title('Reflection Phase vs Frequency (Unwrapped)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', fontsize=10)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Add secondary x-axis for absolute frequency
ax3_top = ax3.twiny()
ax3_top.set_xlim(ax3.get_xlim())
ax3_top.set_xlabel('Frequency offset (GHz)', fontsize=12, color='blue')
ax3_top.tick_params(axis='x', labelcolor='blue')
x_ticks_fsr = ax3.get_xticks()
x_ticks_ghz = x_ticks_fsr * FSR / 1e9
ax3_top.set_xticks(x_ticks_fsr)
ax3_top.set_xticklabels([f'{x:.0f}' for x in x_ticks_ghz])

plt.tight_layout()
plt.show()
print("\nPlot displayed!")

# Now create a detailed plot showing numerator and denominator phases separately
# for one value of R1
fig2, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2, figsize=(14, 10))

R1_example = 0.9
r1 = np.sqrt(R1_example)

# Numerator: -r1 + e^(iφ)
numerator = -r1 + np.exp(1j * phi)
phase_num = np.unwrap(np.angle(numerator))

# Denominator: 1 - r1*e^(iφ)
denominator = 1 - r1 * np.exp(1j * phi)
phase_den = np.unwrap(np.angle(denominator))

# Total phase
phase_total = phase_num - phase_den

# Plot numerator phase
ax4.plot(delta_nu / FSR, np.degrees(phase_num), 'b-', linewidth=2)
ax4.set_ylabel('Phase (degrees)', fontsize=12)
ax4.set_xlabel('Frequency offset (FSR)', fontsize=12)
ax4.set_title(f'Numerator Phase: $-r_1 + e^{{i\\phi}}$ (R₁={R1_example})', 
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot denominator phase
ax5.plot(delta_nu / FSR, np.degrees(phase_den), 'r-', linewidth=2)
ax5.set_ylabel('Phase (degrees)', fontsize=12)
ax5.set_xlabel('Frequency offset (FSR)', fontsize=12)
ax5.set_title(f'Denominator Phase: $1 - r_1 e^{{i\\phi}}$ (R₁={R1_example})', 
              fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot total phase
ax6.plot(delta_nu / FSR, np.degrees(phase_total), 'g-', linewidth=2)
ax6.set_ylabel('Phase (degrees)', fontsize=12)
ax6.set_xlabel('Frequency offset (FSR)', fontsize=12)
ax6.set_title(f'Total Phase (Numerator - Denominator)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot all three together
ax7.plot(delta_nu / FSR, np.degrees(phase_num), 'b-', linewidth=2, label='Numerator', alpha=0.7)
ax7.plot(delta_nu / FSR, np.degrees(phase_den), 'r-', linewidth=2, label='Denominator', alpha=0.7)
ax7.plot(delta_nu / FSR, np.degrees(phase_total), 'g-', linewidth=3, label='Total (Num - Den)')
ax7.set_ylabel('Phase (degrees)', fontsize=12)
ax7.set_xlabel('Frequency offset (FSR)', fontsize=12)
ax7.set_title(f'Phase Decomposition (R₁={R1_example})', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()
