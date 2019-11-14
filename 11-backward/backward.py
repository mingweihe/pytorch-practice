import torch as t
from torch.autograd import Variable as v

# simple gradient
a = v(t.FloatTensor([2, 3]), requires_grad=True)
b = a + 3
c = b * b * 3
out = c.mean()
out.backward()
print('*'*20)
print('------- simple gradient -------')
print('input')
print(a.data)
print('compute result is')
print(out.item())
print('input gradients are')
# As c = (3*(a+3)^2)/2
# so, d(c)/d(a) = 3*(a+3)
# => a.grad.data = [3*(2+3), 3*(3+3)] = [15, 18]
print(a.grad.data)

# backward on non-scalar output
m = v(t.FloatTensor([[2, 3]]), requires_grad=True)
n = v(t.zeros(1, 2))
n[0, 0] = m[0, 0] ** 2
n[0, 1] = m[0, 1] ** 3
n.backward(t.FloatTensor([[1, 1]]))

print('*' * 20)
print('------- non scalar output -------')
print('input')
print(m.data)
print('input gradients are')
# As n0 = m0^2, n1 = m1^3
# so, d(n0)/d(m0) = 2*m0, d(n1)/d(m1) = 3*m1^2
# ==> m.grad.data = [[2*2, 3*3^2]] = [[4, 27]]
print(m.grad.data)

# Jacobian
print('*' * 20)
j = t.zeros(2, 2)
k = v(t.zeros(1, 2))
m.grad.data.zero_()
k[0, 0] = m[0, 0] ** 2 + 3 * m[0, 1]
k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
k.backward(t.FloatTensor([[1, 0]]), retain_graph=True)
j[: 0] = m.grad.data
m.grad.data.zero_()
k.backward(t.FloatTensor([[0, 1]]))
j[:, 1] = m.grad.data
print('Jacobian matrix is')
print(j)