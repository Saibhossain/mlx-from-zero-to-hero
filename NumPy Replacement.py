import mlx.core as mx

a = mx.array([[1.0, 2.0], [3.0, 4.0]])
b = mx.array([[5.0, 6.0], [7.0, 8.0]])

print("Matrix Addition:\n", a + b)
print("Matrix Multiplication:\n", mx.matmul(a, b))