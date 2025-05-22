import numpy as np
import sympy
from sympy import integrate, cos, sin, Symbol

# --- HW8 Problem 1 ---
print("--- HW8 Problem 1 ---")

# Given data
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

num_points = len(x)

# a. Construct the least squares approximation of degree two and compute the error.
print("\na. Least Squares Approximation of Degree Two")
# For a quadratic polynomial of the form y = ax^2 + bx + c, we solve the following normal equations:
# a(Σx_i^2) + b(Σx_i) + c(Σ1) = Σf(x_i)
# a(Σx_i^3) + b(Σx_i^2) + c(Σx_i) = Σx_i f(x_i)
# a(Σx_i^4) + b(Σx_i^3) + c(Σx_i^2) = Σx_i^2 f(x_i)
# Reordered into matrix form as in the provided context:
# [Σx_i^4  Σx_i^3  Σx_i^2] [a] = [Σx_i^2 f(x_i)]
# [Σx_i^3  Σx_i^2  Σx_i  ] [b]   [Σx_i f(x_i) ]
# [Σx_i^2  Σx_i    num_points] [c]   [Σf(x_i)    ]

sum_x = np.sum(x)
sum_x2 = np.sum(x**2)
sum_x3 = np.sum(x**3)
sum_x4 = np.sum(x**4)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2y = np.sum(x**2 * y)

A_matrix_q1a = np.array([
    [sum_x4, sum_x3, sum_x2],
    [sum_x3, sum_x2, sum_x],
    [sum_x2, sum_x, num_points]
])

B_vector_q1a = np.array([sum_x2y, sum_xy, sum_y])

# Solve the system of equations
coefficients_q1a = np.linalg.solve(A_matrix_q1a, B_vector_q1a)
a_quad, b_quad, c_quad = coefficients_q1a

print(f"   Least squares quadratic polynomial: P(x) = {a_quad:.4f}x^2 + {b_quad:.4f}x + {c_quad:.4f}")

# Calculate the error
predicted_y_q1a = a_quad * x**2 + b_quad * x + c_quad
error_q1a = np.sum((y - predicted_y_q1a)**2)
print(f"   Error (Σ[y_i - P(x_i)]^2): {error_q1a:.4f}")

# b. Construct the least squares approximation of the form be^(ax) and compute the error.
print("\nb. Least Squares Approximation of the form be^(ax)")
# Take the natural logarithm of y, linearizing it to ln y = ln b + ax.
# Let Y = ln y, A = a, B = ln b. This becomes a linear form: Y = AX + B.
ln_y = np.log(y)

# Construct the linear system of equations (Y = AX + B)
# [Σx_i^2  Σx_i] [A] = [Σx_i Y_i]
# [Σx_i    num_points] [B]   [ΣY_i    ]
sum_x_lin_q1b = np.sum(x)
sum_x_squared_lin_q1b = np.sum(x**2)
sum_ln_y_q1b = np.sum(ln_y)
sum_x_ln_y_q1b = np.sum(x * ln_y)

A_matrix_q1b = np.array([
    [sum_x_squared_lin_q1b, sum_x_lin_q1b],
    [sum_x_lin_q1b, num_points]
])

B_vector_q1b = np.array([sum_x_ln_y_q1b, sum_ln_y_q1b])

# Solve the system for A (which is 'a_exp') and B (which is 'ln_b_exp')
coefficients_q1b = np.linalg.solve(A_matrix_q1b, B_vector_q1b)
a_exp = coefficients_q1b[0]
ln_b_exp = coefficients_q1b[1]
b_exp = np.exp(ln_b_exp)

print(f"   Least squares approximation y = {b_exp:.4f}e^({a_exp:.4f}x)")

# Calculate the error
predicted_y_q1b = b_exp * np.exp(a_exp * x)
error_q1b = np.sum((y - predicted_y_q1b)**2)
print(f"   Error (Σ[y_i - be^(ax_i)]^2): {error_q1b:.4f}")

# c. Construct the least squares approximation of the form bx^n and compute the error.
print("\nc. Least Squares Approximation of the form bx^n")
# Take the natural logarithm of y, linearizing it to ln y = ln b + n ln x.
# Let Y = ln y, X = ln x, A = n, B = ln b. This becomes a linear form: Y = AX + B.
ln_x = np.log(x)
ln_y_q1c = np.log(y)

# Construct the linear system of equations
# [ΣX_i^2  ΣX_i] [A] = [ΣX_i Y_i]
# [ΣX_i    num_points] [B]   [ΣY_i    ]
sum_ln_x_q1c = np.sum(ln_x)
sum_ln_x_squared_q1c = np.sum(ln_x**2)
sum_ln_y_q1c = np.sum(ln_y_q1c)
sum_ln_x_ln_y_q1c = np.sum(ln_x * ln_y_q1c)

A_matrix_q1c = np.array([
    [sum_ln_x_squared_q1c, sum_ln_x_q1c],
    [sum_ln_x_q1c, num_points]
])

B_vector_q1c = np.array([sum_ln_x_ln_y_q1c, sum_ln_y_q1c])

# Solve the system for n and ln(b)
coefficients_q1c = np.linalg.solve(A_matrix_q1c, B_vector_q1c)
n_power = coefficients_q1c[0]
ln_b_power = coefficients_q1c[1]
b_power = np.exp(ln_b_power)

print(f"   Least squares approximation y = {b_power:.4f}x^({n_power:.4f})")

# Calculate the error
predicted_y_q1c = b_power * (x**n_power)
error_q1c = np.sum((y - predicted_y_q1c)**2)
print(f"   Error (Σ[y_i - bx_i^n]^2): {error_q1c:.4f}")

# --- HW8 Problem 2 ---
print("\n--- HW8 Problem 2 ---")

# 2. Find the least squares polynomial approximation of degree two on the interval [-1,1] for the function f(x)=1/2cos x + 1/4sin 2x.
x_sym = Symbol('x')
f_sym = (1/2) * cos(x_sym) + (1/4) * sin(2 * x_sym)

# For continuous least squares, we find P_n(x) = a_n x^n + ... + a_0 that minimizes E = ∫[f(x) - P_n(x)]^2 dx.
# This is done by solving the normal equations: ∂E/∂a_i = 0, which implies ∫[f(x) - P_n(x)]x^i dx = 0.
# For n=2, we are looking for P_2(x) = a_2 x^2 + a_1 x + a_0.
# The system of equations is:
# ∫[-1,1] (a_2 x^2 + a_1 x + a_0) dx = ∫[-1,1] f(x) dx
# ∫[-1,1] (a_2 x^3 + a_1 x^2 + a_0 x) dx = ∫[-1,1] x f(x) dx
# ∫[-1,1] (a_2 x^4 + a_1 x^3 + a_0 x^2) dx = ∫[-1,1] x^2 f(x) dx

# Calculate the integrals for the right-hand side
integral_f_q2 = integrate(f_sym, (x_sym, -1, 1))
integral_xf_q2 = integrate(x_sym * f_sym, (x_sym, -1, 1))
integral_x2f_q2 = integrate(x_sym**2 * f_sym, (x_sym, -1, 1))

# Construct the normal equations matrix.
# For the interval [-1, 1], integrals of odd powers of x are zero.
# ∫1 dx = 2
# ∫x^2 dx = 2/3
# ∫x^4 dx = 2/5
# The system simplifies to:
# 2a_0 + (2/3)a_2 = ∫f(x)dx
# (2/3)a_1 = ∫xf(x)dx
# (2/3)a_0 + (2/5)a_2 = ∫x^2f(x)dx

# Matrix A
A_matrix_q2 = np.array([
    [2, 0, 2/3],
    [0, 2/3, 0],
    [2/3, 0, 2/5]
])

# Vector B
B_vector_q2 = np.array([float(integral_f_q2), float(integral_xf_q2), float(integral_x2f_q2)])

# Solve the system of equations
coefficients_q2 = np.linalg.solve(A_matrix_q2, B_vector_q2)
a0_poly, a1_poly, a2_poly = coefficients_q2

print(f"   Least squares quadratic polynomial: P_2(x) = {a2_poly:.4f}x^2 + {a1_poly:.4f}x + {a0_poly:.4f}")

# --- HW8 Problem 3 ---
print("\n--- HW8 Problem 3 ---")

# 3. Determine the discrete least squares trigonometric polynomial S_4
# for f(x)=x^2sin x on the interval [0,1] using m=16.

# First, transform the interval [0,1] to [-pi, pi].
# The transformation formula is z_i = pi * [2 * (x_i - c) / (d - c) - 1].
# Here c=0, d=1, so z_i = pi * (2x_i - 1).
# The lecture notes specify z_i = -pi + (i/m)*pi for i = 0 to 2m-1.
m = 16
n_terms = 4 # for S_4

z_points = np.array([-np.pi + (i / m) * np.pi for i in range(2 * m)]) # 2m points (i=0 to 2m-1)

# Convert z_i points back to the original x interval [0,1]
# From z = pi * (2x - 1) => x = (z/pi + 1) / 2
x_original_points = (z_points / np.pi + 1) / 2

# Calculate f(x) = x^2 * sin(x) for the original x points
y_values = x_original_points**2 * np.sin(x_original_points)

# a. Determine the coefficients for S_4
# The coefficients are given by:
# a_0 = (1/m)Σy_i
# a_l = (1/m)Σy_i cos(l*x_i)
# b_l = (1/m)Σy_i sin(l*x_i)
# where x_i here refers to the transformed z_i points in [-pi, pi].

a0_fourier = (1 / m) * np.sum(y_values)
a_coeffs = [0] * (n_terms + 1) # a_0, a_1, ..., a_n
b_coeffs = [0] * (n_terms + 1) # b_1, ..., b_{n-1} (and b_n for completeness, though often zero)

for k in range(1, n_terms): # k from 1 to n-1
    a_coeffs[k] = (1 / m) * np.sum(y_values * np.cos(k * z_points))
    b_coeffs[k] = (1 / m) * np.sum(y_values * np.sin(k * z_points))

# Calculate a_n (a_4 in this case)
a_coeffs[n_terms] = (1 / m) * np.sum(y_values * np.cos(n_terms * z_points))

# For b_n (b_4 in this case), it is typically 0 in the standard Fourier series for real functions.
# Based on the given S_n(x) definition, the sum for sin(kx) goes up to n-1.
# So, b_n is not included in the sum for the coefficients, which means it defaults to 0.
b_coeffs[n_terms] = 0.0 # b_4 is effectively 0 for this series definition

print("\na. Coefficients for the Discrete Least Squares Trigonometric Polynomial S_4:")
print(f"   a_0 = {a0_fourier:.4f}")
for k in range(1, n_terms + 1): # Loop up to n_terms to include a_n
    print(f"   a_{k} = {a_coeffs[k]:.4f}")
for k in range(1, n_terms): # Loop up to n_terms - 1 for b_k based on S_n(x) definition
    print(f"   b_{k} = {b_coeffs[k]:.4f}")
# b_4 is not explicitly part of the S_n(x) sum from the lecture notes definition.

# Define the S_4(z) function (in the transformed z-interval)
def S4_z(z_val):
    s_val = (1/2) * a0_fourier
    s_val += a_coeffs[n_terms] * np.cos(n_terms * z_val) # a_n term
    for k in range(1, n_terms): # sum from k=1 to n-1
        s_val += a_coeffs[k] * np.cos(k * z_val) + b_coeffs[k] * np.sin(k * z_val)
    return s_val

# Define S_4(x) in the original x-interval [0,1]
def S4_x(x_val):
    z_val = np.pi * (2 * x_val - 1)
    return S4_z(z_val)

print("\n   Form of S_4(x) (with coefficients calculated in the z-interval [-pi, pi]):")
print(f"   S_4(z) = 0.5*{a0_fourier:.4f} + {a_coeffs[4]:.4f}cos(4z) + {a_coeffs[1]:.4f}cos(z) + {b_coeffs[1]:.4f}sin(z) + \\")
print(f"             {a_coeffs[2]:.4f}cos(2z) + {b_coeffs[2]:.4f}sin(2z) + {a_coeffs[3]:.4f}cos(3z) + {b_coeffs[3]:.4f}sin(3z)")
print("   where z = pi * (2x - 1)")

# b. Compute ∫S_4(x)dx
# Relationship between x and z: x = (z/pi + 1)/2, so dx = (1/(2pi))dz.
# The integration interval transforms from [0,1] to [-pi, pi].
# ∫[0,1] S_4(x) dx = ∫[-pi,pi] S_4(z) (1/(2pi)) dz
# For a trigonometric polynomial over [-pi, pi], only the constant term's integral is non-zero:
# ∫[-pi,pi] (1/2)a_0 dz = (1/2)a_0 * (2pi) = a_0 * pi
# Therefore, ∫[0,1] S_4(x) dx = (1/(2pi)) * (a_0_fourier * pi) = a_0_fourier / 2

integral_S4_x = a0_fourier / 2
print(f"\nb. ∫S_4(x)dx from 0 to 1 = {integral_S4_x:.4f}")

# c. Compare the integral in part (b) to ∫x^2sin(x)dx.
# Calculate the exact integral
integral_exact_q3c = integrate(x_sym**2 * sin(x_sym), (x_sym, 0, 1))
print(f"\nc. Exact integral ∫x^2sin(x)dx from 0 to 1 = {float(integral_exact_q3c):.4f}")
print(f"   Difference between integrals: {abs(integral_S4_x - float(integral_exact_q3c)):.4f}")

# d. Compute the error E(S_4)
# The error E(S_4) refers to the discrete least squares error, i.e., Σ[y_i - S_4(x_i)]^2.
# We use the original y_values and the computed S4_x values to calculate the error.
predicted_y_s4 = S4_x(x_original_points)
error_S4 = np.sum((y_values - predicted_y_s4)**2)
print(f"\nd. Error E(S_4) = Σ[y_i - S_4(x_i)]^2: {error_S4:.4f}")