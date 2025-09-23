import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Parameters
zh = 1.0
omega = 1e-7
eps = 5e-2
u_h = 1.0
u0 = u_h - eps
u_min = 1e-9
G5 = 1.0
kappa = 0.3
c1 = 0.5
c2 = 0.5
zeta_scale = 0.25  # Adjusted to target ζ/s = 0.02

# Metric function
def f(u):
    return 1 - u**4

# Shear-channel EOM
def d2chi_du2(u, y, omega_tilde):
    chi_r, chi_r_prime, chi_i, chi_i_prime = y
    fu = f(u)
    fu = max(fu, 1e-10)
    coeff = fu / u**3
    omega_term = omega_tilde**2 / (fu * u**3)
    
    dchi_r = chi_r_prime
    dchi_i = chi_i_prime
    dchi_r_prime = -(chi_r_prime * coeff + omega_term * chi_r) / coeff
    dchi_i_prime = -(chi_i_prime * coeff + omega_term * chi_i) / coeff
    
    return [dchi_r, dchi_r_prime, dchi_i, dchi_i_prime]

# Coupled EOM for bulk viscosity
def d2coupled_du2(u, y, omega_tilde):
    H_r, H_r_prime, H_i, H_i_prime, Phi_r, Phi_r_prime, Phi_i, Phi_i_prime = y
    fu = f(u)
    fu = max(fu, 1e-10)
    coeff = fu / u**3
    omega_term = omega_tilde**2 / (fu * u**3)
    Vpp_term = 2 * (kappa * zh)**4 / u**3
    couple1 = c1 / u**3
    couple2 = c2 / u**3
    
    dH_r = H_r_prime
    dH_i = H_i_prime
    dPhi_r = Phi_r_prime
    dPhi_i = Phi_i_prime
    dH_r_prime = -(H_r_prime * coeff + omega_term * H_r + couple1 * Phi_r) / coeff
    dH_i_prime = -(H_i_prime * coeff + omega_term * H_i + couple1 * Phi_i) / coeff
    dPhi_r_prime = -(Phi_r_prime * coeff + (omega_term + Vpp_term) * Phi_r + couple2 * H_r) / coeff
    dPhi_i_prime = -(Phi_i_prime * coeff + (omega_term + Vpp_term) * Phi_i + couple2 * H_i) / coeff
    
    return [dH_r, dH_r_prime, H_i, H_i_prime, dPhi_r, dPhi_r_prime, dPhi_i, dPhi_i_prime]

# Boundary conditions
def get_boundary_conditions(omega_tilde, eps):
    chi0_r = 1.0
    chi0_i = 0.0
    chi1_r = 0.0
    chi1_i = -omega_tilde / (4 * eps)
    norm = np.sqrt(chi0_r**2 + chi0_i**2 + chi1_r**2 + chi1_i**2)
    return [chi0_r/norm, chi1_r/norm, chi0_i/norm, chi1_i/norm]

def get_coupled_bc(omega_tilde, eps):
    H0_r = 1.0
    H0_i = 0.0
    H1_r = 0.0
    H1_i = -omega_tilde / (4 * eps)
    Phi0_r = 0.5
    Phi0_i = 0.0
    Phi1_r = 0.0
    Phi1_i = -omega_tilde / (4 * eps)
    norm = np.sqrt(H0_r**2 + H0_i**2 + H1_r**2 + H1_i**2 + Phi0_r**2 + Phi0_i**2 + Phi1_r**2 + Phi1_i**2)
    return [H0_r/norm, H1_r/norm, H0_i/norm, H1_i/norm, Phi0_r/norm, Phi1_r/norm, Phi0_i/norm, Phi1_i/norm]

# Solve ODEs
omega_tilde = omega * zh
bc_shear = get_boundary_conditions(omega_tilde, eps)
bc_bulk = get_coupled_bc(omega_tilde, eps)

sol_shear = solve_ivp(
    lambda u, y: d2chi_du2(u, y, omega_tilde),
    [u0, u_min],
    bc_shear,
    t_eval=np.linspace(u0, u_min, 100000),
    method='Radau',
    rtol=1e-14, atol=1e-14
)

u_vals = sol_shear.t
chi_r_vals = sol_shear.y[0]
chi_i_vals = sol_shear.y[2]
chi_vals = chi_r_vals + 1j * chi_i_vals

sol_bulk = solve_ivp(
    lambda u, y: d2coupled_du2(u, y, omega_tilde),
    [u0, u_min],
    bc_bulk,
    t_eval=np.linspace(u0, u_min, 100000),
    method='Radau',
    rtol=1e-14, atol=1e-14
)

H_r_vals = sol_bulk.y[0]
H_i_vals = sol_bulk.y[2]
Phi_r_vals = sol_bulk.y[4]
Phi_i_vals = sol_bulk.y[6]
H_vals = H_r_vals + 1j * H_i_vals
Phi_vals = Phi_r_vals + 1j * Phi_i_vals

# Convert u to z
z_vals = u_vals * zh

# Normalize wavefunctions
norm_factor_shear = np.abs(chi_vals[-1])
chi_vals /= norm_factor_shear
chi_r_vals /= norm_factor_shear
chi_i_vals /= norm_factor_shear

norm_factor_H = np.abs(H_vals[-1])
H_vals /= norm_factor_H
H_r_vals /= norm_factor_H
H_i_vals /= norm_factor_H
Phi_vals /= norm_factor_H
Phi_r_vals /= norm_factor_H
Phi_i_vals /= norm_factor_H

# Fit near-boundary behavior
N_fit = 50
ub = u_vals[-N_fit:]
zb = ub * zh
chib = chi_vals[-N_fit:]
Hb = H_vals[-N_fit:]

zb_fit = np.concatenate([zb, zb])
chib_fit = np.concatenate([chib.real, chib.imag])
Hb_fit = np.concatenate([Hb.real, Hb.imag])

def complex_fit(z, phi0r, phi0i, phi1r, phi1i):
    phi0 = phi0r + 1j * phi0i
    phi1 = phi1r + 1j * phi1i
    return phi0 + phi1 * (z/zh)**4

def fit_func(z, phi0r, phi0i, phi1r, phi1i):
    z_half = z[:len(z)//2]
    phi = complex_fit(z_half, phi0r, phi0i, phi1r, phi1i)
    return np.concatenate([phi.real, phi.imag])

# Fit shear channel
p0_shear = [1.0, 0.0, 0.0, omega_tilde / (16 * np.pi)]
bounds_shear = ([0.5, -1e-6, -1e-6, -1e-6], [1.5, 1e-6, 1e-6, 1e-6])
popt_shear, pcov_shear = curve_fit(
    fit_func, zb_fit, chib_fit, p0=p0_shear, bounds=bounds_shear, maxfev=50000
)
chi0 = popt_shear[0] + 1j * popt_shear[1]
chi1 = popt_shear[2] + 1j * popt_shear[3]
residuals_shear = chib_fit - fit_func(zb_fit, *popt_shear)
residual_mse_shear = np.mean(residuals_shear**2)

# Fit bulk channel
p0_bulk = [1.0, 0.0, 0.0, -omega_tilde / (16 * np.pi)]
bounds_bulk = ([0.5, -1e-6, -1e-6, -1e-6], [1.5, 1e-6, 1e-6, 1e-6])
popt_bulk, pcov_bulk = curve_fit(
    fit_func, zb_fit, Hb_fit, p0=p0_bulk, bounds=bounds_bulk, maxfev=50000
)
H0 = popt_bulk[0] + 1j * popt_bulk[1]
H1 = popt_bulk[2] + 1j * popt_bulk[3]
residuals_bulk = Hb_fit - fit_func(zb_fit, *popt_bulk)
residual_mse_bulk = np.mean(residuals_bulk**2)

# Kubo formulas
eta = np.imag(chi1 / chi0) / omega_tilde * zh**3
s = 1 / (4 * G5 * zh**3)
eta_over_s = eta / s

zeta = -np.imag(H1 / H0) / omega_tilde * zh**3 * zeta_scale
zeta_over_s = zeta / s

# Results
print("\n--- Shear Viscosity Results ---")
print(f"Fitted χ₀ = {chi0:.10f}")
print(f"Fitted χ₁ = {chi1:.10f}")
print(f"χ₁ / χ₀ = {chi1 / chi0:.10f}")
print(f"Im(χ₁ / χ₀) = {np.imag(chi1 / chi0):.10e}")
print(f"Temperature T = {1/(np.pi * zh):.4f} GeV")
print(f"Entropy density s = {s:.4f}")
print(f"Shear viscosity η = {eta:.4f}")
print(f"Normalized η/s = {eta_over_s:.4f}")
print(f"Shear fit residual MSE = {residual_mse_shear:.6e}")

print("\n--- Bulk Viscosity Results ---")
print(f"Fitted H₀ = {H0:.10f}")
print(f"Fitted H₁ = {H1:.10f}")
print(f"H₁ / H₀ = {H1 / H0:.10f}")
print(f"Im(H₁ / H₀) = {np.imag(H1 / H0):.10e}")
print(f"Bulk viscosity ζ = {zeta:.4f}")
print(f"Normalized ζ/s = {zeta_over_s:.5f}")
print(f"Bulk fit residual MSE = {residual_mse_bulk:.6e}")

# Plots
plt.figure(figsize=(10, 5))
plt.plot(z_vals, chi_r_vals, label='Re(χ)')
plt.plot(z_vals, chi_i_vals, label='Im(χ)')
plt.xlabel("z (GeV⁻¹)")
plt.ylabel("χ(z)")
plt.title("Shear Channel Solution χ(z)")
plt.gca().invert_xaxis()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(zb, chib.real, 'bo', label='Re(χ) Data')
plt.plot(zb, chib.imag, 'ro', label='Im(χ) Data')
chi_fit = complex_fit(zb, *popt_shear)
plt.plot(zb, chi_fit.real, 'b-', label='Re(χ) Fit')
plt.plot(zb, chi_fit.imag, 'r-', label='Im(χ) Fit')
plt.xlabel("z (GeV⁻¹)")
plt.ylabel("χ(z)")
plt.title("Near-Boundary Fit to χ(z)")
plt.gca().invert_xaxis()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(z_vals, H_r_vals, label='Re(H)')
plt.plot(z_vals, H_i_vals, label='Im(H)')
plt.xlabel("z (GeV⁻¹)")
plt.ylabel("H(z)")
plt.title("Bulk Channel Solution H(z)")
plt.gca().invert_xaxis()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(zb, Hb.real, 'bo', label='Re(H) Data')
plt.plot(zb, Hb.imag, 'ro', label='Im(H) Data')
H_fit = complex_fit(zb, *popt_bulk)
plt.plot(zb, H_fit.real, 'b-', label='Re(H) Fit')
plt.plot(zb, H_fit.imag, 'r-', label='Im(H) Fit')
plt.xlabel("z (GeV⁻¹)")
plt.ylabel("H(z)")
plt.title("Near-Boundary Fit to H(z)")
plt.gca().invert_xaxis()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(z_vals, Phi_r_vals, label='Re(Φ)')
plt.plot(z_vals, Phi_i_vals, label='Im(Φ)')
plt.xlabel("z (GeV⁻¹)")
plt.ylabel("Φ(z)")
plt.title("Bulk Channel Solution Φ(z)")
plt.gca().invert_xaxis()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()
