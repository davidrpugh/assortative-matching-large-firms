def trim_data(xs, ys, x_value):
	x_new = []
	y_new = []
	for i in range(len(xs)):
		if xs[i]<= x_value:
			x_new.append(xs[i])
			y_new.append(ys[i])

	return np.asarray(x_new), np.asarray(y_new)
