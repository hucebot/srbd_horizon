import casadi as ca
import matplotlib.pyplot as plt


H = ca.Sparsity.from_file("debug_gnsqp_H.mtx")

plt.matshow(H)
plt.title("Debug view of H gnsqp matrices")
plt.legend()


A = ca.Sparsity.from_file("debug_gnsqp_A.mtx")

plt.matshow(A)
plt.title("Debug view of A gnsqp matrices")
plt.legend()

plt.show()
