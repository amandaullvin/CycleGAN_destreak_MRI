import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer


# FIXME ------------------------------------------
import random
import torch
# manualSeed = random.randint(1, 10000) # fix seed
manualSeed = 1984
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# FIXME ------------------------------------------  




opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0


gen_iterations = 0
for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):
    epoch_start_time = time.time()

    data_iter = iter(dataset)
    i = 0

    while i < len(dataset):
        # Frozen and unfrozen within model.bacward_G
        # model.freeze_discriminators(False) # will train discriminators

        # train the discriminators Diters times
        if (gen_iterations < 25 and not opt.skip_warmup) or (gen_iterations % 500 == 0 and gen_iterations > 0):
            diters = 100
        else:
            diters = opt.diter

        j = 0
        while j < diters and i < len(dataset):
            iter_start_time = time.time()
            j += 1
            data = data_iter.next() # sample data real_A and real_B
            model.set_input(data)
            i += 1
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)    

            # actual training:
            model.forward()
            model.optimize_parameters_D()

            # print losses to console
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                #if opt.display_id > 0:
                #    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
            
        # if we consume the dataset during training D, just start new epoch without training G
        if i >= len(dataset):
            break

        # now train the generators

        # Frozen and unfrozen within model.bacward_G
        data = data_iter.next() # sample data real_A and real_B
        model.set_input(data)
        i += 1
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)

        # actual training:
        model.forward()
        model.optimize_parameters_G()
        gen_iterations += 1


        # update visdom only in G updates:
        if gen_iterations % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)


    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

    if epoch > opt.nepoch:
        model.update_learning_rate()
