import pandas as pd
import numpy as np
import sympy
import matplotlib.pyplot as plt


class Models:
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.m = len(self.data)
        self.y = self.data['deaths'].to_numpy()
        self.x = np.arange(self.m)
        self._var = sympy.Symbol('x')
        self._cubic = self.equation(self._poly_equation(3))
        self._quad = self.equation(self._poly_equation(2))
        self._lag = self._lagrange()

    def _poly_equation(self, n):
        A = np.zeros((n+1, n+1))
        B = np.zeros(n+1)
        A[0][0] = self.m
        for i in range(n+1):
            for j in range(n+1):
                if i == 0 and j == 0:
                    continue
                A[i, j] = np.sum(self.x**(i+j))
            B[i] = np.sum(self.x**i * self.y)

        eq = np.dot(np.linalg.inv(A), B)
        return eq

    def _extract_date(self, input):
        for i in range(self.m):
            if input == self.data.iloc[i]['date']:
                return self.data.iloc[i]['id']
        raise NameError('not found')

    def _lagrange(self):
        poly = 0
        for i in range(len(self.data)):
            temp = 1
            for j in range(len(self.data)):
                if i == j:
                    continue
                top = (self._var - self.x[j])
                bottom = (self.x[i] - self.x[j])
                temp *= top/bottom
            poly += temp * self.y[i]
        return poly

    def predict_cubic(self, time):
        x = self._extract_date(time)
        return self._cubic.subs(self._var, x)

    def predict_quad(self, time):
        x = self._extract_date(time)
        return self._quad.subs(self._var, x)

    def relative_error(self, measured, real):
        return (measured - real)/real

    def predict_lagrange(self, time):
        return self._lag.subs(self._var, self._extract_date(time))

    def error(self):
        cubic = 0
        lag = 0
        quad = 0
        for i in range(self.m):
            val = self.y[i]
            cubic += self.relative_error(self.predict_cubic(self.data.iloc[i]['date']), val)
            lag += self.relative_error(self.predict_lagrange(self.data.iloc[i]['date']), val)
            quad += self.relative_error(self.predict_quad(self.data.iloc[i]['date']), val)
        return cubic, lag, quad

    def graph(self, model):
        lam_x = sympy.lambdify(self._var, model, modules=['numpy'])
        x_vals = np.arange(self.m)
        y_vals = lam_x(x_vals)
        plt.ylim(0,2400)
        plt.xlim(0,50)
        plt.scatter(self.x, self.y)
        plt.plot(x_vals, y_vals, '.r-')
        plt.show()

    def graph_all(self):
        lam_x_quad = sympy.lambdify(self._var, self._cubic, modules=['numpy'])
        lam_x_cub = sympy.lambdify(self._var, self._quad, modules=['numpy'])
        lam_x_lag = sympy.lambdify(self._var, self._lag, modules=['numpy'])
        x_vals = np.arange(self.m)
        y1 = lam_x_cub(x_vals)
        y2 = lam_x_quad(x_vals)
        y3 = lam_x_lag(x_vals)
        plt.ylim(0, 2400)
        plt.xlim(0, 50)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].scatter(self.x, self.y)
        axs[0, 0].set_title('data points')
        axs[1, 0].scatter(self.x, self.y)
        axs[1, 0].plot(x_vals, y1)
        axs[1, 0].set_title('cubic poly')
        axs[0, 1].scatter(self.x, self.y)
        axs[0, 1].plot(x_vals, y2)
        axs[0, 1].set_title('quad poly')
        axs[1, 1].scatter(self.x, self.y)
        axs[1, 1].plot(x_vals, y3)
        axs[1, 1].set_title('lag')
        fig.tight_layout()
        plt.show()

    def equation(self, eq):
        poly = 0
        for i in range(len(eq)):
            poly += eq[i] * self._var ** i
        return poly

    def cubic_equation(self):
        return self._cubic

    def quad_equation(self):
        return self._quad

    def lag_equation(self):
        return self._lag

foo = Models('data.csv')
x1 = '3/20'
y1 = 65
x2 = '4/1'
y2 = 1021
x3 = '4/20'
y3 = 1837

print('x:     {}              {}              {}'.format(x1, x2, x3))
print('y:     {}                {}             {}'.format(y1, y2, y3))
print('cubic  {} {} {}'.format(foo.predict_cubic(x1), foo.predict_cubic(x2), foo.predict_cubic(x3)))
print('quad   {} {} {}'.format(foo.predict_quad(x1), foo.predict_quad(x2), foo.predict_quad(x3)))
print('lag    {}                {}             {}'.format(foo.predict_lagrange(x1), foo.predict_lagrange(x2), foo.predict_lagrange(x3)))

foo.graph_all()
print(foo.error())
