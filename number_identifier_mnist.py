import numpy as np
from math import e
from sys import argv

from os import environ
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
pygame.init()

# simple networks work fast enough
# from cProfile import Profile
# from pstats import Stats, SortKey
# def profile_func(func, *args):
# 	with Profile() as prof:
# 		func(*args)
# 	stats = Stats(prof)
# 	stats.sort_stats(SortKey.TIME)
# 	stats.dump_stats(filename=f"{randint(0,100000)}.prof")

# HOW TO USE
# - LMB for white (to write with); RMB for black (to erase with); mouse wheel to change brush size; R on keyboard to reset board
# NOTE: the image is size-normalised and centred around "centre" of image, please notice this it took a lot of effort to get working
# NOTER: this script is labelled "mnist" because it only properly works with networks trained on MNIST! might be cause of the unique image preprocessing done to EMNIST, will try fixing that later

# TODOS
# - put code into classes and make it neater? rushed this a ton so funcs and magic numbers are all over the place

# don't change!
IMG_W = IMG_H = 20
TRUE_W = TRUE_H = 28
PIXELS = TRUE_W * TRUE_H

# can change!
RESIZE_RATIO = 15
PG_W, PG_H = IMG_W * RESIZE_RATIO, IMG_H * RESIZE_RATIO
BRUSH_SCALE = 5
BRUSH_MIN, BRUSH_MAX = 1, 50

datasets = {
	"class"   : (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122),
	"balanced": (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116),
	# PROBLEM WITH LETTERS! for some reason the mapping doesn't have a value for index 0, so you'll have to add 1 to the neurone label
	"letters" : (0, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90),
	"merge"   : (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116),
	"digits"  : tuple(range(48,58)),
	"emnist mnist": tuple(range(48,58)),
	"mnist"   : tuple(range(48,58)),
}
# need to manually change this for each network fed into this script! maybe I should add "network_used" in the network files, but too late for now...
CURRENT_DATASET = "mnist"

def map_emnist_char(index):
	return chr(datasets[CURRENT_DATASET][index])

scr = pygame.display.set_mode((PG_W, PG_H))
draw_surface = pygame.Surface((PG_W, PG_H))
resized_scr = pygame.Surface((IMG_W, IMG_H))
padded_scr = pygame.Surface((TRUE_W, TRUE_H))
icon = pygame.image.load("stuff\\bigbrain.png")
pygame.display.set_icon(icon)
pygame.display.set_caption("Number: ")
clock = pygame.time.Clock()

# absolutely hate my implementation of size normalisation but heyyy it works
crop_left, crop_top, crop_right, crop_bot = None, None, None, None

def update_brushes():
	global brush_white, brush_black, brushrect
	brush_white = pygame.transform.smoothscale_by(brush_white_og, BRUSH_SCALE)
	brush_black = pygame.transform.smoothscale_by(brush_black_og, BRUSH_SCALE)
	brushrect = brush_white.get_rect()

# use brushwl or brushbl for smoother brushes.. or make your own
brush_white_og = pygame.image.load("stuff\\brushw.png")
brush_black_og = pygame.image.load("stuff\\brushb.png")
update_brushes()

# make 2 2d arrays corresponding to img, initialised with each pixel's x or y values
y_distances = np.resize(np.array([i for i in range(IMG_W)]), (IMG_W, IMG_H))
x_distances = np.resize(np.array([i for i in range(IMG_W)]), (IMG_H, IMG_W)).transpose()

def calc_centre_of_mass(img):
	'''calculate centre of mass of image'''
	x_moments = x_distances * img
	y_moments = y_distances * img
	total_weight = img.sum()
	return (x_moments.sum()/total_weight, y_moments.sum()/total_weight) if total_weight != 0 else (0,0)


class Network:
	def __init__(self, filename, fontsize):
		with np.load(filename) as data:
			layers = len(data)//2
			self.LAYER_NUM = layers + 1
			self.NEURONE_NUM = [PIXELS] * self.LAYER_NUM
			self.neurones = [[Neurone(0, input_neurone_i, 0) for input_neurone_i in range(PIXELS)]]
			
			for layer_i_m1 in range(layers):
				layer_i = layer_i_m1 + 1
				current_weights = data[f"arr_{layer_i_m1}"]

				self.NEURONE_NUM[layer_i] = current_weights.shape[0]
				# initialise neurones but with 0s as the weights and biases
				self.neurones.append([Neurone(layer_i, neurone_i, layer_i == self.LAYER_NUM-1) for neurone_i in range(self.NEURONE_NUM[layer_i])])

				for current_neurone_i, current_neurone_weights in enumerate(current_weights):
					self.neurones[layer_i][current_neurone_i].weights = current_neurone_weights

				layer_i_m1 += layers
				current_biases = data[f"arr_{layer_i_m1}"]
				for current_neurone_i, current_neurone_bias in enumerate(current_biases):
					self.neurones[layer_i][current_neurone_i].bias = current_neurone_bias

		self.numfont = Label("bahnschrift", fontsize, (PG_W, 0), (0,0,0))

	def activation_func(self, z):
		# TODO - change to ReLU later
		return 1/(1 + e**(-z))

	def activation_derivative(self, z):
		'''calculates delA/delZ, since A = act_func(z)'''
		# also change to ReLU
		e_to_z = e**z
		return e_to_z/(e_to_z + 1)**2

	def get_weight_matrix(self, layer_i):
		'''creates a 2d array of width prev_neurone_num and height current_neurone_num; represents all weights leading to each neurone in current_layer'''
		return np.array([current_layer_neurone.weights for current_layer_neurone in self.neurones[layer_i]])

	def analyse_img(self, img_buffer):
		#if len(img_buffer) != self.NEURONE_NUM[0]: raise Exception("input dont match bruh")

		# normalising 8-bit int pixel data into floats
		norm_img_data = img_buffer/255
		for i, input_neurone in enumerate(self.neurones[0]):
			input_neurone.activation = norm_img_data[i]

		for layer_i in range(1, self.LAYER_NUM):
			# collect all previous layer neurone activations, weights, and biases as float-type ndarrays, and sum together
			# prev_activations will be a 1d array of length prev_neurone_num; current_biases of length current_neurone_num
			prev_activations = np.array([prev_neurone.activation for prev_neurone in self.neurones[layer_i - 1]])
			current_biases = np.array([current_neurone.bias for current_neurone in self.neurones[layer_i]])

			weights = self.get_weight_matrix(layer_i)

			# 1d array of length current_neurone_num
			current_zs = np.matmul(weights, prev_activations) + current_biases
			current_activations = self.activation_func(current_zs)
			current_delA_by_delZs = self.activation_derivative(current_zs)

			for current_neurone_i, current_neurone in enumerate(self.neurones[layer_i]):
				# log activations and derivatives for each neurone to help in training
				current_neurone.activation = current_activations[current_neurone_i]
				current_neurone.delA_by_delZ = current_delA_by_delZs[current_neurone_i]

		guess = map_emnist_char(self.neurones[-1][current_activations.argmax()].label)
		pygame.display.set_caption(f"Number: {guess}")
		sureness = current_activations[current_activations.argmax()]
		sureness = 200 * (sureness if sureness <= 1 else 1)
		self.numfont.render(guess, (0, sureness + 55, 0))


class Neurone:
	def __init__(self, layer, index, is_output=False):
		self.activation = None
		self.layer = layer

		if layer != 0:
			self.bias = 0
			self.weights = 0
			# explicitly defining label instead of directly using index will make adapting this to NIST dataset (with latin characters as well) easier
			self.label = index if is_output else None


class Label:
	def __init__(self, fontname, size, topright, bg_colour):
		self.font = pygame.font.SysFont(fontname, size)
		self.topright = topright
		self.bg = bg_colour

	def render(self, text, colour):
		self.img = self.font.render(str(text), True, colour)
		self.rect = self.img.get_rect()
		self.rect.topright = self.topright
		scr.blit(self.img, self.rect)


net = Network(argv[1] if len(argv) > 1 else input("Filename: "), 30)
# net = Network(argv[1] if len(argv) > 1 else "2 class 14.7 200 2730.npz", 30)
net.numfont.render("", (0,0,0))

while 1:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			exit()

		elif event.type == pygame.MOUSEWHEEL:
			BRUSH_SCALE += event.y
			BRUSH_SCALE = BRUSH_MIN if BRUSH_SCALE < BRUSH_MIN else BRUSH_MAX if BRUSH_SCALE > BRUSH_MAX else BRUSH_SCALE
			update_brushes()

		elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION:
			scr.blit(draw_surface, (0,0))
			brushrect.center = event.pos
			mouse_buttons = pygame.mouse.get_pressed()

			drawing = True
			if mouse_buttons[0]:	brush = brush_white
			elif mouse_buttons[2]:	brush = brush_black
			else:					drawing = False

			if drawing:
				draw_surface.blit(brush, brushrect)

				# something malicious is brewing...
				if crop_left  is None or brushrect.left   < crop_left:   crop_left = brushrect.left if brushrect.left > 0 else 0
				if crop_right is None or brushrect.right  > crop_right: crop_right = brushrect.right if brushrect.right < PG_W else PG_W
				if crop_top   is None or brushrect.top    < crop_top:     crop_top = brushrect.top if brushrect.top > 0 else 0
				if crop_bot   is None or brushrect.bottom > crop_bot:     crop_bot = brushrect.bottom if brushrect.bottom < PG_H else PG_H

				# this line makes the program not work for non-square images, but eh
				crop_w = max(crop_right - crop_left, crop_bot - crop_top)
				crop_img = pygame.Surface((crop_w, crop_w))
				crop_img.blit(draw_surface, (0,0), (crop_left, crop_top, crop_w, crop_w))

				pygame.transform.smoothscale(crop_img, (IMG_W, IMG_H), resized_scr)

				px = pygame.surfarray.array_red(resized_scr)
				centre = calc_centre_of_mass(px)

				padded_scr.fill((0,0,0))
				# place 20x20 image in 28x28 padded image such that CoM is at the centre of padded img
				padded_scr.blit(resized_scr, (TRUE_W//2 - centre[0], TRUE_H//2 - centre[1]))
			
			scr.blit(padded_scr, (0,0))
			net.analyse_img(pygame.surfarray.array_red(padded_scr).transpose().ravel())
			scr.blit(brush_white, brushrect)
			
			pygame.display.flip()
			clock.tick(300)

		elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
			draw_surface.fill((0,0,0))
			scr.fill((0,0,0))
			pygame.display.set_caption("Number: ")
			pygame.display.flip()
			crop_left, crop_top, crop_right, crop_bot = None, None, None, None