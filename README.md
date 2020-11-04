# convstacks

(wip)

A simple library for stacking convolutions in pytorch. Compositions of convolutions are themselves convolutions. For example, 
stacks of increasingly dilated convolutions underpin the Wavenet model with efficient long-range correlations.

---
training 

x0, .... x5 (n = 6)
with k = 2

x2 = f(x0, x1)
...
x5 = f(x3, x4)


x0, ..., xn-1 inputs
y0, ....,yn-1 outputs
model as sequence of convolutional layers w/left padding

k effective kernel size:
	xk  = f(x0, ..., xk-1)		
	...
	xn-1 = f(xn-1-k, ..., xn-2)


graphically lth output gets tied to lth input and preceding k - 1 inputs

yl = f(xl-k+1,...xl)

y0	y1  y2  y3  ... yn-1
*	*	*	*	*	*

 (convolutional layers)

*	*   *   *	*	*
x0  x1  x2  x3  ... xn-1

So match model by matching output to input as y1 = x2, y2 = x3 etc,
so that xl+1 = f(xl-k+1,...xl), as desired.

so train model with loss function to match yl to xl+1.
in loss function keep only full relationships: ignore first k - 1 outputs, 
ignore last output, ignore last input.

loss = sum_i (i = k-1, n-2) el_loss(y_i, x_i+1)

so looks like

	    xk  ...        xn-1
		*	*	*	*	*

 (convolutional layers)

*	*   *   *	*	*	*
x0  .. xk-1  ....      xn-2



