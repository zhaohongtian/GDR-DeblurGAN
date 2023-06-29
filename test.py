import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from multiprocessing import freeze_support
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from util.metrics import SSIM
from PIL import Image

def test(opt,dataset, model, visualizer):
	# create website
	web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
	# test
	avgPSNR = 0.0
	avgSSIM = 0.0
	counter = 0
	t = 0.0
	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		counter = i
		iter_start_time = time.time()
		model.set_input(data)
		model.test()
		visuals = model.get_current_visuals()
		avgPSNR += PSNR(visuals['fake_B'],visuals['real_A'])
		# pilFake = Image.fromarray(visuals['fake_B'])
		# pilReal = Image.fromarray(visuals['real_A'])
		# avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)

		img_path = model.get_image_paths()
		t = t + (time.time() - iter_start_time)
		print('process image... %s' % img_path)
		visualizer.save_images(webpage, visuals, img_path)
		

	#
	avgPSNR /= counter
	# avgSSIM /= counter
	print('PSNR = %f'%(avgPSNR))   #, SSIM = %f   , avgSSIM
	time_average = t/1111
	print('time_average = %f' % time_average)
	webpage.save()

if __name__ == '__main__':
	freeze_support()
	opt = TestOptions().parse()
	opt.nThreads = 1  # test code only supports nThreads = 1
	opt.batchSize = 1  # test code only supports batchSize = 1
	opt.serial_batches = True  # no shuffle
	opt.no_flip = True  # no flip
	#opt.dataroot = 'path_gopro/to/data/B'
	opt.dataroot = '/home/wd/Rosa/DeblurGAN-Densenet/path_gopro/to/data/A'
	# opt.dataroot = 'path_kohler/to/data/A'
	#opt.dataroot = 'path_real/to/data/A'
	opt.model = 'test'
	opt.dataset_mode = 'single'
	opt.learn_residual = True
	opt.resize_or_crop = None


	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	model = create_model(opt)
	visualizer = Visualizer(opt)
	test(opt, dataset, model, visualizer)