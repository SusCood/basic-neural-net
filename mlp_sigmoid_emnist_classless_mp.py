from random import randint, random, shuffle
import numpy as np
from math import e
from sys import argv, stdout
from time import time, perf_counter

from multiprocessing import Pool, Manager, cpu_count
from queue import Empty as QEmpty

# from cProfile import Profile
# from pstats import Stats, SortKey

# TODOS
# - graph the avg percent correct over time in the data file
# - ^do this for many different settings, multiple lines on same graph! see what the best combination of neurones and BATCH_SIZE is
# - make ReLU work
# - get rid of neurone class for performance? and other optimisations
# - add a learning accelerator variable when adding derivative, maybe even make it proportional to % somehow
#
# - PARALLELISATION!!!
# will have to check how to share memory from one parent to 6 or so child processes
# definitely TRY OUT USING POOL *IN* the batch loop (starting a new task pool for every batch)
# maybe try JoinableQueue in subprocess (task_ready() on each batch completion)

# DEBUG TODOS
# - make a smaller version of the process communication part that keeps messing up, and see what makes it work

class DatasetInfo:
	'''glorified dictionary'''
	def __init__(self, filepaths, output_map):
		self.trainimg, self.trainlabel, self.testimg, self.testlabel = filepaths
		self.output_map = output_map

datasets = {
	"class"   : DatasetInfo(("emnist\\class\\emnist-byclass-train-images-idx3-ubyte", "emnist\\class\\emnist-byclass-train-labels-idx1-ubyte", "emnist\\class\\emnist-byclass-test-images-idx3-ubyte", "emnist\\class\\emnist-byclass-test-labels-idx1-ubyte"), (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122)),
	"balanced": DatasetInfo(("emnist\\balanced\\emnist-balanced-train-images-idx3-ubyte", "emnist\\balanced\\emnist-balanced-train-labels-idx1-ubyte" ,"emnist\\balanced\\emnist-balanced-test-images-idx3-ubyte", "emnist\\balanced\\emnist-balanced-test-labels-idx1-ubyte"), (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116)),
	# PROBLEM WITH LETTERS! for some reason the mapping doesn't have a value for index 0, so you'll have to add 1 to the neurone label
	"letters" : DatasetInfo(("emnist\\letters\\emnist-letters-train-images-idx3-ubyte", "emnist\\letters\\emnist-letters-train-labels-idx1-ubyte", "emnist\\letters\\emnist-letters-test-images-idx3-ubyte", "emnist\\letters\\emnist-letters-test-labels-idx1-ubyte"), (0, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90)),
	"merge"   : DatasetInfo(("emnist\\merge\\emnist-bymerge-train-images-idx3-ubyte", "emnist\\merge\\emnist-bymerge-train-labels-idx1-ubyte", "emnist\\merge\\emnist-bymerge-test-images-idx3-ubyte", "emnist\\merge\\emnist-bymerge-test-labels-idx1-ubyte"), (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116)),
	"digits"  : DatasetInfo(("emnist\\digits\\emnist-digits-train-images-idx3-ubyte", "emnist\\digits\\emnist-digits-train-labels-idx1-ubyte", "emnist\\digits\\emnist-digits-test-images-idx3-ubyte", "emnist\\digits\\emnist-digits-test-labels-idx1-ubyte"), tuple(range(48,58))),
	"emnist mnist": DatasetInfo(("emnist\\mnist\\emnist-mnist-train-images-idx3-ubyte", "emnist\\mnist\\emnist-mnist-train-labels-idx1-ubyte", "emnist\\mnist\\emnist-mnist-test-images-idx3-ubyte", "emnist\\mnist\\emnist-mnist-test-labels-idx1-ubyte"), tuple(range(48,58))),
	"mnist"   : DatasetInfo(("mnist\\train-images.idx3-ubyte", "mnist\\train-labels.idx1-ubyte", "mnist\\t10k-images.idx3-ubyte", "mnist\\t10k-labels.idx1-ubyte"), tuple(range(48,58))),
}


def getbytes(start_offset, byte_no, cast_to_int=False, use_test=False):
	'''returns array of bytes. cast_to_int is used when converting bytes to integer'''
	data = t_img_data if use_test else img_data
	return int.from_bytes(data[start_offset:start_offset + byte_no], "big") if cast_to_int else data[start_offset:start_offset + byte_no]


def get_ran_array(size=None):
	'''returns single/ndarray of random values from a std distro'''
	return rng_gen.standard_normal(size) * RNG_STD_DEV


def map_emnist_char(index):
	return chr(datasets[CURRENT_DATASET].output_map[index])


def pg_check_quit(network, save_on_exit=True):
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			if save_on_exit and network.SAVE: network.save()
			pygame.quit()
			exit()


def profile_func(func, *args):
	with Profile() as prof:
		func(*args)

	stats = Stats(prof)
	stats.sort_stats(SortKey.TIME)
	stats.dump_stats(filename=net.get_filename(".prof"))


def printf(printstr):
	'''needed to print stuff when using multiprocessing'''
	print(printstr, flush=True)
	stdout.flush()


def get_img(data_index=-1, use_test=False):
	'''returns a random image's 8bit data and its label from dataset, or image+label at specified index if data_index != -1'''
	if data_index == -1:
		data_index = randint(0, (T_IMG_NUM if use_test else IMG_NUM) - 1)

	img_offset = 16 + PIXELS * data_index
	label_offset = 8 + data_index

	img = np.frombuffer(getbytes(img_offset, PIXELS, use_test=use_test), dtype=np.uint8)
	# for some reason, EMNIST data needs to be transposed, while the original MNIST data does not
	if CURRENT_DATASET != "mnist":
		img.shape = HEIGHT, WIDTH
		img = img.transpose().flatten()
	return img, (t_label_data if use_test else label_data)[label_offset]


def activation_func(z):
	# TODO - change to ReLU later
	return 1/(1 + e**(-z))

	# return np.tanh(z)
	
	# z[z < 0] *= self.ReLU_LEAK
	# return z

def activation_derivative(z):
	'''calculates delA/delZ, since A = act_func(z)'''
	e_to_z = e**z
	return e_to_z/(e_to_z + 1)**2

	# return 1/np.cosh(z)**2

	# z[z >= 0] = 1
	# z[z < 0] = self.ReLU_LEAK
	# return z


class Network:
	def __init__(self, neurones, draw = False, save = False, batch_size = 6 * 200, draw_l1 = True):
		self.neurone_biases, self.neurone_weights = [None], [None]
		if isinstance(neurones, tuple):
			self.LAYER_NUM = len(neurones) 	# 4
			self.NEURONE_NUM = neurones 	# (784,32,16,10)

			for layer_i in range(1, self.LAYER_NUM):
				# randomise weights and biases for all neurones
				self.neurone_biases.append(get_ran_array(self.NEURONE_NUM[layer_i]))
				# neurone_weights = list of LAYER_NUM ndarrays, each having dimensions of current_layer_neurones x previous_layer_neurones (rows x columns)
				self.neurone_weights.append(get_ran_array( (self.NEURONE_NUM[layer_i], self.NEURONE_NUM[layer_i - 1]) ))
		else:
			# if neurones is a string, load with neurones as filename instead
			self.load(neurones)

		#self.labels = range(neurones[-1]); will be using output neurone indices instead
		self.neurone_actns = [np.zeros(layer_neurone_num) for layer_neurone_num in self.NEURONE_NUM]
		self.neurone_dA_by_dZs = [None] + [np.zeros(layer_neurone_num) for layer_neurone_num in self.NEURONE_NUM[1:]]

		# best pick a batch_size that's a multiple of CPUS
		self.BATCH_SIZE = batch_size
		self.DRAW = draw
		self.SAVE = save
		# self.ReLU_LEAK = leak_factor
		self.img_indices = list(range(IMG_NUM))
		self.img_list_index = 0

		#if self.DRAW:
		self.PG_W, self.PG_H = 1000, 800
		self.PG_W_FRAC = 0.8
		self.PG_NET_W = int(self.PG_W * self.PG_W_FRAC)
		self.PG_CIRCLE_W = 6
		self.PG_LINE_W = 2
		self.PG_POS_LINE_HUE = 20
		self.PG_NEG_LINE_HUE = 200
		self.PG_BG_COLOUR = (190,) * 3
		# if draw_l1 is false, first set of weights (usually the largest) will never be drawn in training/testing, considerably speeding up drawing
		self.DRAW_LAYER_1 = draw_l1

		self.pg_scr = pygame.display.set_mode((self.PG_W, self.PG_H))
		self.pg_circle_layer = pygame.Surface((self.PG_W, self.PG_H), flags=pygame.SRCALPHA)
		pygame.display.set_caption("Neural Network!")
		pygame.display.set_icon(icon)
		self.labelfont = Label("bahnschrift", self.PG_H//10, ((0, 180, 0), (180, 0, 0)), ((self.PG_W + self.PG_NET_W)//2, 2 * self.PG_H//3), self.PG_BG_COLOUR)
		self.pg_colour = pygame.Color(0,0,0)
		self.neurone_x_coords = np.linspace(self.PG_NET_W, 0, num = self.LAYER_NUM, endpoint = False)[::-1]

	def get_weight_matrix(self, layer_i):
		'''creates a 2d array of width prev_neurone_num and height current_neurone_num; represents all weights leading to each neurone in current_layer'''
		return self.neurone_weights[layer_i]

	def shuffle_images(self):
		'''reset counter and reshuffle list if run out of data, and repeat until all the batches are exhausted'''
		shuffle(self.img_indices)
		self.img_list_index = 0

	def get_filename(self, extension):
		'''returns a string to be used for a unique filename (for network data, images, performance profiles). not an attribute coz self.percent changes'''
		return f"n-{','.join(str(num) for num in net.NEURONE_NUM)} d-{CURRENT_DATASET} p-{net.percent} b-{self.BATCH_SIZE} c-{CPUS} {randint(0,100000)}{extension}"

	def create_empty_weights_biases(self):
		'''for layers from the 2nd onward, make a list of 2d weight matrices leading to that layer, and 1d bias matrices of that layer'''
		# initialised w/ 1 None to hold nonexistent weights for the first layer; makes indexing easier
		del_weights, del_biases = [None], [None]
		for layer_i in range(1, self.LAYER_NUM):
			del_weights.append(np.zeros( (self.NEURONE_NUM[layer_i], self.NEURONE_NUM[layer_i - 1]) ))
			del_biases.append(np.zeros(self.NEURONE_NUM[layer_i]))
		return del_weights, del_biases

	def calc_cost(self, one_img_data, label):
		'''calculates cost. cost = sigmoid(weights_matrix x prev_activations_vector + bias_vector)'''
		# if len(one_img_data) != self.NEURONE_NUM[0]: raise Exception("input dont match bruh")
		self.neurone_actns[0] = one_img_data/255

		for layer_i in range(1, self.LAYER_NUM):
			current_zs = np.matmul(self.neurone_weights[layer_i], self.neurone_actns[layer_i - 1]) + self.neurone_biases[layer_i]
			self.neurone_actns[layer_i] = activation_func(current_zs)
			self.neurone_dA_by_dZs[layer_i] = activation_derivative(current_zs)
		
		costs = self.neurone_actns[-1]**2
		costs[label] = (self.neurone_actns[-1][label] - 1)**2

		return np.sum(costs), self.neurone_actns[-1].argmax()


	def train(self, num_of_batches):
		self.shuffle_images()
		print(f"Random success rate is ~{round(100/len(datasets[CURRENT_DATASET].output_map))}%")

		with (
				Pool(CPUS) as pool,
				# for some reason, Manager.Queue() is needed? tasks dont even start if using other Queues
				Manager() as mgr,
			):
			# for sending initialisation data to child processes
			init_data_q = mgr.Queue()
			# and for receiving del data from child processes
			del_data_q = mgr.Queue()
			data = (
				init_data_q,
				del_data_q,
			)

			tasks = pool.map_async(train_subprocess, (data, ) * CPUS)

			for batch_i in range(num_of_batches):
				start_t = time()

				# will be adding to these counts everytime a child process returns with data
				total_cost = 0
				correct = 0
				del_weights, del_biases = self.create_empty_weights_biases()

				if len(self.img_indices) - self.img_list_index < self.BATCH_SIZE: self.shuffle_images()
				# after each child process returns, add self.BATCH_SIZE//CPUS to self.img_list_index!

				for cpu in range(CPUS):
					index_start = self.img_list_index + self.BATCH_SIZE//CPUS * cpu
					# for each process, put network data + BATCH_SIZE/CPU length slice of indices into init_q
					# self.neurone_dels aren't needed
					init_data_q.put((
						self.neurone_weights,
						self.neurone_biases,
						self.img_indices[index_start : index_start + self.BATCH_SIZE//CPUS],
						self.BATCH_SIZE//CPUS,
					))

				self.img_list_index += self.BATCH_SIZE

				subprocesses_left = CPUS
				while subprocesses_left:
					# could use to terminate all processes too first?
					pg_check_quit(self)

					try:
						sub_del_weights, sub_del_biases, sub_costs, sub_corrects = del_data_q.get(timeout=QUEUE_TIMEOUT)
						total_cost += sub_costs
						correct += sub_corrects
						for layer_i in range(1, self.LAYER_NUM):
							del_weights[layer_i] += sub_del_weights[layer_i]
							del_biases[layer_i] += sub_del_biases[layer_i]
						subprocesses_left -= 1
					except QEmpty:
						# need to keep pumping pg events, so i can't block main process forever
						continue	

				for layer_i in range(1, self.LAYER_NUM):
					# to get average of derivatives, unaffected by BATCH_SIZE			
					del_weights[layer_i] /= self.BATCH_SIZE
					del_biases[layer_i] /= self.BATCH_SIZE

					self.neurone_weights[layer_i] -= del_weights[layer_i]
					self.neurone_biases[layer_i] -= del_biases[layer_i]

				if self.DRAW or batch_i == num_of_batches - 1:
					# draw with last used image; does take a lot of time tho
					self.draw(img, label, current_correct)

				self.percent = round(correct/self.BATCH_SIZE * 100, 1)
				time_taken = round(time() - start_t, 4)
				printf(f"Batch {batch_i+1}: {time_taken} s, average cost {round(total_cost/self.BATCH_SIZE, 4)}, {correct}/{self.BATCH_SIZE} correct ({self.percent}%); {round((batch_i + 1)/num_of_batches * 100, 1)}% complete (~{round(time_taken/60 * (num_of_batches - batch_i - 1))} min left)")

		# no clue why I have to do this, even after using a context manager...
		del pool, mgr
		# see if you don't have to manually terminate all the other processes before this


	def draw(self, img, label, correct_output, network_changing=True):
		# NEED TO FIXXXXXXXXXX!
		if network_changing: self.pg_scr.fill(self.PG_BG_COLOUR)
		self.pg_circle_layer.fill((255, 255, 255, 0))

		pg_buffer = np.zeros(len(img) * 3, dtype=np.uint8)
		# need to convert grayscale 8-bit to RGB 24-bit, coz for SOME REASON pygame doesn't support PIL's "L" mode :|
		for i in range(len(pg_buffer)):
			pg_buffer[i] = img[i//3]
		pg_img = pygame.image.frombuffer(pg_buffer, (WIDTH, HEIGHT), "RGB")
		pg_img = pygame.transform.scale_by(pg_img, 3)	# TODO: make this responsive maybe idfk this really isn't sth i should waste my time on goddamn
		pg_rec = pg_img.get_rect()
		pg_rec.center = (self.PG_W + self.PG_NET_W)//2, self.PG_H//3
		self.pg_scr.blit(pg_img, pg_rec)

		self.labelfont.render(map_emnist_char(label), correct_output, self.pg_scr, network_changing)	

		for layer_i in range(self.LAYER_NUM):
			current_neurone_coords_list = [(self.neurone_x_coords[layer_i], y) for y in np.linspace(0, self.PG_H, num = self.NEURONE_NUM[layer_i]+2)[1:-1]]

			if network_changing and layer_i != 0 and (self.DRAW_LAYER_1 or layer_i != 1):
				for current_neurone_i, current_coords in enumerate(current_neurone_coords_list):
					current_weights = self.neurone_weights[layer_i][current_neurone_i]
					# map [-inf, inf] weights to [0, 100] colour val
					line_colour_vals = 100/(1 + e**(-0.55 * abs(current_weights)))
					for prev_neurone_i, prev_coords in enumerate(prev_neurone_coords_list):
						self.pg_colour.hsva = self.PG_NEG_LINE_HUE if current_weights[prev_neurone_i] < 0 else self.PG_POS_LINE_HUE, 100, line_colour_vals[prev_neurone_i], 100

						pygame.draw.line(self.pg_scr, self.pg_colour, prev_coords, current_coords, self.PG_LINE_W)
						#pygame.display.update(line_rect)

			for current_neurone_i, current_coords in enumerate(current_neurone_coords_list):
				if layer_i != 0:
					current_bias = self.neurone_biases[layer_i][current_neurone_i]
					circle_colour = (255, 80, 80, 255) if current_bias < 0 else (80, 255, 80, 255)
					bias_width = round(5 * abs(current_bias)**(1/3))
					pygame.draw.circle(self.pg_circle_layer, circle_colour, current_coords, self.PG_CIRCLE_W + bias_width)

				#circle_colour = 0, 0, round(self.neurones[layer_i][current_neurone_i].activation * 255), 255
				self.pg_colour.hsla = 240, 100, self.neurone_actns[layer_i][current_neurone_i] * 100, 100
				pygame.draw.circle(self.pg_circle_layer, self.pg_colour, current_coords, (self.PG_CIRCLE_W + bias_width) if (layer_i == self.LAYER_NUM-1 and current_neurone_i == label) else self.PG_CIRCLE_W)

			prev_neurone_coords_list = current_neurone_coords_list
			#pygame.display.flip()

		self.pg_scr.blit(self.pg_circle_layer, (0,0))
		pygame.display.flip()


	def save(self, only_pg_screen=False):
		if not only_pg_screen:
			# npz data format: layer1_weights, ..., layerN_weights, layer1_biases, ..., layerN_biases
			np.savez(self.get_filename(""), *(self.neurone_weights[1:] + self.neurone_biases[1:]))

		pygame.image.save(self.pg_scr, self.get_filename(".png"))


	def load(self, filename):
		with np.load(filename) as data:
			layers = len(data)//2
			self.LAYER_NUM = layers + 1
			self.NEURONE_NUM = [PIXELS] * self.LAYER_NUM
			
			for layer_i_minus1 in range(layers):
				# might change naming system to not rely on arr_n naming, but that'd require converting my old dataset files
				self.NEURONE_NUM[layer_i_minus1 + 1] = data[f"arr_{layer_i_minus1}"].shape[0]

				self.neurone_weights.append(data[f"arr_{layer_i_minus1}"])
				self.neurone_biases.append(data[f"arr_{layer_i_minus1 + layers}"])


	def test(self, test_num, framerate=0):
		total_correct = 0
		clock = pygame.time.Clock()
		print(f"{test_num} tests being run...")

		for test_i in range(test_num):
			start_t = perf_counter()
			pg_check_quit(self)

			img, label = get_img(data_index= -1 if test_num != T_IMG_NUM else test_i, use_test=True)
			cost, max_actn_i = self.calc_cost(img, label)
			current_correct = int(max_actn_i == label)
			total_correct += current_correct

			# commenting this out since there'll be a ton of prints
			#print(f"Test {test_i}: cost {round(cost, 4)}, time {round(perf_counter() - start_t, 8)}")
			# draw() with network_changing draws entire screen w/ weights and all; network_changing off only updates important parts
			if self.DRAW or test_i == test_num - 1: self.draw(img, label, current_correct, network_changing = test_i==0)
			#self.save(True)
			clock.tick(framerate)

		self.percent = round(total_correct/test_num * 100, 2)
		print(f"{total_correct}/{test_num} correct ({self.percent}%)")
		#self.save(True)


class Label:
	def __init__(self, fontname, size, colours, centre, bg_colour):
		self.font = pygame.font.SysFont(fontname, size)
		self.colour_right, self.colour_wrong = colours
		self.centre = centre
		self.bg_colour = bg_colour

	def render(self, text, correct_output, pg_surface, network_changing):
		self.img = self.font.render(str(text), True, self.colour_right if correct_output else self.colour_wrong)
		self.rect = self.img.get_rect()
		self.rect.center = self.centre

		if not network_changing:
			# will need to clear only parts of screen
			blank_img = pygame.Surface((int(self.rect.width * 1.4), int(self.rect.height * 1.4)))
			blank_rect = blank_img.get_rect()
			blank_rect.center = self.centre
			blank_img.fill(self.bg_colour)
			pg_surface.blit(blank_img, blank_rect)

		pg_surface.blit(self.img, self.rect)


def train_subprocess(queues):
	'''main function for the trainer child processes, where index and network data is loaded, backpropagation performed, and del_data sent back'''
	# need a way to kill subprocesses when needed, too
	printf("started")
	init_data_q, del_data_q = queues

	# might need to print("{exception}", flush=True) to log exceptions from child processes
	running = True
	while running:
		try:
			(
				neurone_weights,
				neurone_biases,
				img_indices,
				sub_batch_size,
			) = init_data_q.get()
			# no timeout needed, as the subprocess will wait either way until it gets init data, and processes should be killed from main process automatically
			
			neurone_actns = [np.zeros(neurone_weights[1].shape[1])] + [np.zeros(layer_weights.shape[0]) for layer_weights in neurone_weights[1:]]
			neurone_dA_by_dZs, del_biases = ([None] + [np.zeros(layer_biases.shape) for layer_biases in neurone_biases[1:]], ) * 2
			del_weights = [None] + [np.zeros(layer_weights.shape) for layer_weights in neurone_weights[1:]]

			layer_num = len(neurone_biases)
			total_cost, correct = 0, 0

			for image_num in range(sub_batch_size):
				img, label = get_img(img_indices[image_num])

				# modified from Network.calc_cost(), didn't want to add another unnecessary func call
				neurone_actns[0] = img/255
				for layer_i in range(1, layer_num):
					current_zs = np.matmul(neurone_weights[layer_i], neurone_actns[layer_i - 1]) + neurone_biases[layer_i]
					neurone_actns[layer_i] = activation_func(current_zs)
					neurone_dA_by_dZs[layer_i] = activation_derivative(current_zs)
				
				cost = neurone_actns[-1]**2
				cost[label] = (neurone_actns[-1][label] - 1)**2
				cost = np.sum(cost)
				max_actn_i = neurone_actns[-1].argmax()

				total_cost += cost
				# using index of highest activation neurone as the label itself
				current_correct = int(max_actn_i == label)
				correct += current_correct

				# can directly calc delC/delA at last layer: 2(A - e) (= 2A if not_label, 2A - 2)
				dC_by_dAs = 2 * neurone_actns[-1]
				dC_by_dAs[label] -= 2

				# iterate from last to 2nd layer
				for layer_i in range(layer_num - 1, 0, -1):
					if layer_i != layer_num - 1:
						dC_by_dAs = np.matmul(neurone_weights[layer_i + 1].transpose(), dC_by_dZs)

					dC_by_dZs = neurone_dA_by_dZs[layer_i] * dC_by_dAs
					del_biases[layer_i] += dC_by_dZs
					for current_neurone_i in range(neurone_biases[layer_i].size):
						del_weights[layer_i][current_neurone_i] += neurone_actns[layer_i - 1] * dC_by_dZs[current_neurone_i]
			
			del_data_q.put((del_weights, del_biases, total_cost, correct))
		except Exception as e:
			printf(f"EXCEPTION! {e}")
			exit()


CURRENT_DATASET = "mnist"
# open only training data for all processes (parent and children)
with (
		open(datasets[CURRENT_DATASET].trainimg,  "rb") as img_f,
		open(datasets[CURRENT_DATASET].trainlabel,"rb") as label_f,
	):
	img_data, label_data = img_f.read(), label_f.read()

IMG_NUM = getbytes(4, 4, True)
HEIGHT, WIDTH = getbytes(8, 4, True), getbytes(12, 4, True)
PIXELS = HEIGHT * WIDTH
# timeout ensures pygame events can be pumped even when the main process is waiting for data
QUEUE_TIMEOUT = 0.1
CPUS = cpu_count()	# can change!

if __name__ == "__main__":
	from os import environ
	environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
	import pygame
	pygame.init()

	CURRENT_DATASET = "mnist"
	# open testing data only for main process
	with (
			open(datasets[CURRENT_DATASET].testimg,   "rb") as t_img_f,
			open(datasets[CURRENT_DATASET].testlabel, "rb") as t_label_f,
		):
		t_img_data, t_label_data = t_img_f.read(), t_label_f.read()

	T_IMG_NUM = getbytes(4, 4, True, True)
	rng_gen = np.random.default_rng()
	RNG_STD_DEV = 2
	icon = pygame.image.load("stuff\\bigbrain.png")

	net = Network((argv[1] if len(argv) > 1 else (PIXELS, 24, 16, len(datasets[CURRENT_DATASET].output_map))), draw=True, save=True, batch_size=200, draw_l1=False)
	#net = Network("3 89.0 9489.npz", draw=True, save=False, batch_size=500, draw_l1=True)

	#try:
	#profile_func(net.train, 5000)
	net.train(1000)
	net.test(T_IMG_NUM)
	#except Exception as e:
	#	print(e)

	if net.SAVE: net.save()

	while 1:
		pg_check_quit(net, False)