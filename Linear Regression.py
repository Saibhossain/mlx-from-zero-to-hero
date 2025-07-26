import numpy as  np
import mlx.core as mx
import mlx.optimizers as optim

X = mx.array(np.random.rand(100, 1) * 10)  # Feature
y = 3 * X + 5 + mx.array(np.random.randn(100, 1))  # Target

print(X,y)
class LinearRegression:
    def __init__(self):
        self.w = mx.zeros((1, 1))
        self.b = mx.zeros((1,))

    def __call__(self, x):
        return x @ self.w + self.b

model = LinearRegression()

def mse(y_pred,y_true):
    return mx.mean((y_pred-y_true)**2)

optimizer = optim.SGD([model.w, model.b], lr=0.01)

for epoch in range(1000):
    def loss_fn():
        preds = model(X)
        return mse(preds, y)
    loss, grads = mx.value_and_grad(loss_fn)()
    optimizer.update(grads)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
