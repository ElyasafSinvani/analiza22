import math


def Trapezoidal_Rule():
	def fx(x):
		"""
		:param x: the x pram that you wan to get his y
		:return:  the y of the x, it's a Point values (x,y)
		"""
		return math.sin(x)

	P = {}  # are dict of prameters
	upper_bound = math.pi  # the upper x in integral
	bottom_block = 0  # the bottom x in integral
	result = 0  # val to return
	# choose your n, how much splits the val's in your line.
	if (upper_bound - bottom_block) < 0:
		n = 4
	else:
		n = upper_bound - bottom_block

	h = ((upper_bound - bottom_block) / n)
	step = upper_bound / n
	for i in range(int(n)):
		P[i] = (h / 2) * (fx(i * step) + fx((i + 1) * step))

	values = P.values()
	result = sum(values)
	return result

print(f'Trapez calc {Trapezoidal_Rule()}')






