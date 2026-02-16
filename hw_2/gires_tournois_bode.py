import numpy as np
import matplotlib.pyplot as plt

# Round-trip phase array (x-axis)
# Scan from 0 to 4π (two complete FSRs)
phi = np.linspace(0, 4*np.pi, 2000)
phi_deg = np.degrees(phi)

# Different R1 values to plot
R1_values = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

# Create Bode plot figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

colors = plt.cm.viridis(np.linspace(0, 1, len(R1_values)))

print("=== Gires-Tournois Bode Analysis ===\n")

for i, R1 in enumerate(R1_values):
    r1 = np.sqrt(R1)  # amplitude reflectivity
    r2 = 1.0  # R2 = 100%
    
    # Complex reflectivity: (-r1 + e^(iφ)) / (1 - r1*e^(iφ))
    numerator = -r1 + np.exp(1j * phi)
    denominator = 1 - r1 * np.exp(1j * phi)
    rho = numerator / denominator
    
    # Magnitude (should be 1)
    magnitude = np.abs(rho)
    magnitude_dB = 20 * np.log10(magnitude)
    
    # Phase
    theta = np.angle(rho)  # reflection phase in radians
    theta_unwrapped = np.unwrap(theta)
    theta_deg = np.degrees(theta_unwrapped)
    
    # Plot magnitude (Bode magnitude plot)
    ax1.plot(phi_deg, magnitude_dB, color=colors[i], 
             label=f'$R_1 = {R1}$', linewidth=2)
    
    # Plot phase (Bode phase plot)
    ax2.plot(phi_deg, theta_deg, color=colors[i], 
             label=f'$R_1 = {R1}$', linewidth=2.5)
    
    # Calculate some key points
    # At resonance (φ = 2πm): maximum slope
    phi_res = np.array([0, 2*np.pi, 4*np.pi])
    for phi_r in phi_res:
        idx = np.argmin(np.abs(phi - phi_r))
        print(f"R1 = {R1:.2f}, φ = {phi_r/np.pi:.1f}π: θ = {theta_deg[idx]:.1f}°")

# Format magnitude plot
ax1.set_ylabel('Magnitude (dB)', fontsize=14)
ax1.set_xlabel('Round-trip phase $\\phi$ (degrees)', fontsize=12)
ax1.set_title('Bode Magnitude Plot: Gires-Tournois Reflection', 
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='best', fontsize=10)
ax1.set_ylim([-0.1, 0.1])
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)

# Add vertical lines at resonances
for m in range(3):
    ax1.axvline(x=m*360, color='red', linestyle=':', alpha=0.3, linewidth=1)

# Format phase plot
ax2.set_ylabel('Reflection Phase $\\theta$ (degrees)', fontsize=14)
ax2.set_xlabel('Round-trip Phase $\\phi$ (degrees)', fontsize=12)
ax2.set_title('Bode Phase Plot: $\\theta(\\phi)$ for Gires-Tournois', 
              fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='best', fontsize=11)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Add vertical lines at resonances
for m in range(3):
    ax2.axvline(x=m*360, color='red', linestyle=':', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.show()
print("\nBode plot displayed!")

# Create a second figure showing phase slope (group delay)
fig2, ax3 = plt.subplots(1, 1, figsize=(14, 6))

for i, R1 in enumerate(R1_values):
    r1 = np.sqrt(R1)
    
    numerator = -r1 + np.exp(1j * phi)
    denominator = 1 - r1 * np.exp(1j * phi)
    rho = numerator / denominator
    
    theta = np.unwrap(np.angle(rho))
    
    # Calculate derivative: dθ/dφ (phase slope, related to group delay)
    dtheta_dphi = np.gradient(theta, phi)
    
    ax3.plot(phi_deg, dtheta_dphi, color=colors[i], 
             label=f'$R_1 = {R1}$', linewidth=2.5)

ax3.set_ylabel('Phase Slope $d\\theta/d\\phi$', fontsize=14)
ax3.set_xlabel('Round-trip Phase $\\phi$ (degrees)', fontsize=12)
ax3.set_title('Group Delay: $d\\theta/d\\phi$ vs $\\phi$ (Related to Dispersion)', 
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', fontsize=11)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Add vertical lines at resonances
for m in range(3):
    ax3.axvline(x=m*360, color='red', linestyle=':', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.show()
print("Group delay plot displayed!")

# Create detailed analytical plot for one R1 value
fig3, ax4 = plt.subplots(1, 1, figsize=(14, 6))

R1_example = 0.9
r1 = np.sqrt(R1_example)

# Analytical expression for phase
# For Gires-Tournois: θ = -2 * arctan((1-r1²)sin(φ) / (2r1 - (1+r1²)cos(φ)))
# Or equivalently from the complex reflection coefficient

numerator = -r1 + np.exp(1j * phi)
denominator = 1 - r1 * np.exp(1j * phi)
rho = numerator / denominator
theta = np.unwrap(np.angle(rho))
theta_deg = np.degrees(theta)

ax4.plot(phi_deg, theta_deg, 'b-', linewidth=3, label='Reflection Phase $\\theta$')

# Also show the linear approximation near each resonance
# Near resonance φ ≈ 2πm: θ ≈ -2πm + constant
for m in range(3):
    phi_res = 2*np.pi*m
    phi_local = np.linspace(phi_res - np.pi/4, phi_res + np.pi/4, 100)
    
    # Linear approximation (only valid very close to resonance for high finesse)
    # The slope at resonance is approximately -2/(1-r1) for high r1
    slope = -2 / (1 - r1)
    theta_approx = slope * (phi_local - phi_res) - 360*m
    
    ax4.plot(np.degrees(phi_local), theta_approx, 'r--', linewidth=2, 
             alpha=0.7, label='Linear approx.' if m == 0 else '')

ax4.set_ylabel('Reflection Phase $\\theta$ (degrees)', fontsize=14)
ax4.set_xlabel('Round-trip Phase $\\phi$ (degrees)', fontsize=12)
ax4.set_title(f'Bode Phase Plot with Linear Approximation (R₁ = {R1_example})', 
              fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best', fontsize=12)

# Add resonance markers
for m in range(3):
    ax4.axvline(x=m*360, color='red', linestyle=':', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.show()
