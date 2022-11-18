import numpy as np
from matplotlib import pyplot as pl, colors
from sys import argv
from random import randint

# FILENAMES = argv[1:] if len(argv) > 1 else [input("Filename: ")]
FILENAMES = argv[1:] if len(argv) > 1 else ["3 mnist 93.4 300 3746.npz"]
# increasing bins makes graph rougher, opposite of R_AVG_RANGE in the other grapher scripts
BINS = 200
HUES = np.linspace(0, 1, len(FILENAMES), endpoint=False)

USE_RELATIVE_FREQ = True
SHOW_BIASES = True
SHOW_WEIGHTS = True
if not SHOW_BIASES and not SHOW_WEIGHTS: exit()

# i manually configure this script to plot either each layer's weights/biases separately, or add them together
# no need to make this easily controllable, not interesting enough

for i, filename in enumerate(FILENAMES):
	filename_short = filename[-filename[::-1].find("\\"):-4]
	with np.load(filename) as data:
		layers = len(data)//2
		weights, biases = np.array([]), np.array([])
		# can choose to either add up all layers' weights and biases, or specific layers if there's enough data (code that in bruv)
		for layer_i in range(layers):
			weights = np.append(weights, data[f"arr_{layer_i}"].flatten())
			biases = np.append(biases, data[f"arr_{layer_i + layers}"].flatten())

		wfreq, wbins = np.histogram(weights, bins=BINS)
		bfreq, bbins = np.histogram(biases, bins=BINS//20)

		if USE_RELATIVE_FREQ:
			wfreq = wfreq/wfreq.max()
			bfreq = bfreq/bfreq.max()

		if SHOW_WEIGHTS: pl.plot(wbins[:-1], wfreq, color=colors.hsv_to_rgb((HUES[i], 1, 0.9)), label="WEIGHT - " + filename_short)
		if SHOW_BIASES: pl.plot(bbins[:-1], bfreq, color=colors.hsv_to_rgb((HUES[i], 1, 0.5)), label="BIAS - " + filename_short)

pl.title("Weights and biases distribution")
pl.xlabel("Weight/bias" if SHOW_WEIGHTS and SHOW_BIASES else "Weight" if SHOW_WEIGHTS else "Bias")
pl.ylabel("Relative frequency" if USE_RELATIVE_FREQ else "Frequency")
pl.grid(True)
pl.legend(loc="best")

pl.show()