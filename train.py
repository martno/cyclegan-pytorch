import click
import os.path
import torch
import torch.nn as nn
from torch.autograd import Variable
import csv
import random
import numpy as np
import itertools
from os.path import join as join_path
import collections
import skimage.io
import natural.number

import models
import model_utils
import constants as c
import image_utils
import utils
from utils import check, Params, Timer, ensure_dir_exists


FIELD_NAMES = ['step', 'generators_loss', 'discr_a_loss', 'discr_b_loss', 'total_loss', 'duration']


@click.command()
@click.option('--dataset_a', type=click.Path(exists=True, dir_okay=True), help='Path to dataset A')
@click.option('--dataset_b', type=click.Path(exists=True, dir_okay=True), help='Path to dataset B')
@click.option('--use_cuda/--no_cuda', default=False, show_default=True)
@click.option('--checkpoint_path', default='checkpoint', show_default=True, help='Checkpoint path')
@click.option('--save_step', default=100, show_default=True, type=click.IntRange(min=1), help='Save every `--save_step` step')
@click.option('--test_step', default=100, show_default=True, type=click.IntRange(min=1), help='Test every `--test_step` step')
@click.option('--batch_size', default=1, show_default=True, type=click.IntRange(min=1), help='Batch size')
@click.option('--image_pool_size', default=50, show_default=True, type=click.IntRange(min=1), help='Image pool size')
@click.option('--adam_beta1', default=0.5, show_default=True, type=float, help="Adam optimizer's beta1 param", callback=check(lambda x: 0 <= x <= 1))
@click.option('--adam_beta2', default=0.999, show_default=True, type=float, help="Adam optimizer's beta2 param", callback=check(lambda x: 0 <= x <= 1))
@click.option('--gen_learning_rate', default=2e-4, show_default=True, type=float, help="Learning rate for generators")
@click.option('--discr_learning_rate', default=1e-4, show_default=True, type=float, help="Learning rate for discriminators")
@click.option('--log_filename', default='log.csv', show_default=True, help='Log filename')
@click.option('--debug_path', default='debug', show_default=True, help='Folder to save debug images')
def train(**kwargs):
    params = Params(kwargs)

    print('Params:')
    params.pretty_print()
    print()

    use_cuda = params.use_cuda
    if use_cuda:
        assert torch.cuda.is_available()

    with Timer('Initializing'):
        a_image_generator = create_image_generator(params.dataset_a)
        b_image_generator = create_image_generator(params.dataset_b)

        gen_a_to_b, gen_b_to_a, discr_a, discr_b = load_models_for_training(params.checkpoint_path)

        print('#weights in gen_a_to_b:', natural.number.number(model_utils.compute_num_weights(gen_a_to_b)))
        print('#weights in gen_b_to_a:', natural.number.number(model_utils.compute_num_weights(gen_b_to_a)))
        print('#weights in discr_a:', natural.number.number(model_utils.compute_num_weights(discr_a)))
        print('#weights in discr_b:', natural.number.number(model_utils.compute_num_weights(discr_b)))

        if use_cuda:
            gen_a_to_b.cuda()
            gen_b_to_a.cuda()
            discr_a.cuda()
            discr_b.cuda()

        betas = (params.adam_beta1, params.adam_beta2)
        optimizer_generators = torch.optim.Adam(params=itertools.chain(gen_a_to_b.parameters(), gen_b_to_a.parameters()),
                                                lr=params.gen_learning_rate, betas=betas)
        optimizer_discr_a = torch.optim.Adam(params=discr_a.parameters(), lr=params.discr_learning_rate, betas=betas)
        optimizer_discr_b = torch.optim.Adam(params=discr_b.parameters(), lr=params.discr_learning_rate, betas=betas)

        cycle_criterion = nn.L1Loss()
        discr_criterion = nn.MSELoss()

        one_array = torch.ones((params.batch_size, 1, 30, 30))  # Has the same size as the output of discr_a and discr_b
        if use_cuda:
            one_array = one_array.cuda()
        one_array = Variable(one_array, requires_grad=False)

        zero_array = torch.zeros((params.batch_size, 1, 30, 30))
        if use_cuda:
            zero_array = zero_array.cuda()
        zero_array = Variable(zero_array, requires_grad=False)

        a_fake_image_pool = image_utils.ImagePool(params.image_pool_size)
        b_fake_image_pool = image_utils.ImagePool(params.image_pool_size)

    header = '\t'.join(FIELD_NAMES)
    print(header)

    # Train:

    with open(params.log_filename, 'w') as csvfile:
        dict_writer = csv.DictWriter(csvfile, FIELD_NAMES)
        dict_writer.writeheader()

        for i in itertools.count():
            timer = Timer('train step', verbose=False)

            with timer:
                a = generate_batch_variable(a_image_generator, use_cuda, params.batch_size)
                b = generate_batch_variable(b_image_generator, use_cuda, params.batch_size)

                generators_loss = models.compute_generators_loss(gen_a_to_b, gen_b_to_a, discr_a, discr_b, a, b,
                                                                 cycle_criterion, discr_criterion, one_array,
                                                                 a_fake_image_pool, b_fake_image_pool)
                optimize(optimizer_generators, generators_loss)

                discr_a_loss = models.compute_discr_loss(discr_a, a, a_fake_image_pool, discr_criterion, zero_array, one_array)
                optimize(optimizer_discr_a, discr_a_loss)

                discr_b_loss = models.compute_discr_loss(discr_b, b, b_fake_image_pool, discr_criterion, zero_array, one_array)
                optimize(optimizer_discr_b, discr_b_loss)

            row = collections.OrderedDict()
            row['step'] = str(i)
            row['generators_loss'] = float_to_string(generators_loss)
            row['discr_a_loss'] = float_to_string(discr_a_loss)
            row['discr_b_loss'] = float_to_string(discr_b_loss)
            row['total_loss'] = float_to_string(generators_loss + discr_a_loss + discr_b_loss)
            row['duration'] = '{0:.2f}s'.format(timer.get_duration())
            dict_writer.writerow(row)
            print('\t'.join(row.values()))

            if i % params.save_step == 0 and i > 0:
                with Timer('Saving models'):
                    ensure_dir_exists(params.checkpoint_path)

                    torch.save(gen_a_to_b.state_dict(), join_path(params.checkpoint_path, c.A_TO_B_GEN_DIR))
                    torch.save(gen_b_to_a.state_dict(), join_path(params.checkpoint_path, c.B_TO_A_GEN_DIR))

                    torch.save(discr_a.state_dict(), join_path(params.checkpoint_path, c.A_DISCR_DIR))
                    torch.save(discr_b.state_dict(), join_path(params.checkpoint_path, c.B_DISCR_DIR))

                print(header)

            if i % params.test_step == 0 and i > 0:
                ensure_dir_exists(params.debug_path)
                with Timer('Creating debug images'):
                    a, b, b_fake, a_fake = create_debug_images(a, b, gen_a_to_b, gen_b_to_a, params.use_cuda)

                    a_filepath = join_path(params.debug_path, '{}-a.jpg'.format(i))
                    skimage.io.imsave(a_filepath, a)

                    b_filepath = join_path(params.debug_path, '{}-b.jpg'.format(i))
                    skimage.io.imsave(b_filepath, b)

                    b_fake_filepath = join_path(params.debug_path, '{}-a-to-b.jpg'.format(i))
                    skimage.io.imsave(b_fake_filepath, b_fake)

                    a_fake_filepath = join_path(params.debug_path, '{}-b-to-a.jpg'.format(i))
                    skimage.io.imsave(a_fake_filepath, a_fake)


def load_models_for_training(checkpoint_path):
    gen_a_to_b = models.GeneratorNet()
    gen_b_to_a = models.GeneratorNet()

    discr_a = models.DiscriminatorNet()
    discr_b = models.DiscriminatorNet()

    if os.path.isdir(checkpoint_path):
        with Timer('Loading from path: ' + checkpoint_path):
            gen_a_to_b.load_state_dict(torch.load(join_path(checkpoint_path, c.A_TO_B_GEN_DIR)))
            gen_b_to_a.load_state_dict(torch.load(join_path(checkpoint_path, c.B_TO_A_GEN_DIR)))

            discr_a.load_state_dict(torch.load(join_path(checkpoint_path, c.A_DISCR_DIR)))
            discr_b.load_state_dict(torch.load(join_path(checkpoint_path, c.B_DISCR_DIR)))

    else:
        print('No checkpoint found')

    return gen_a_to_b, gen_b_to_a, discr_a, discr_b


def create_image_generator(path):
    filenames = utils.listdir(path, extensions=('.jpg', '.png'))

    while True:
        random.shuffle(filenames)

        for filename in filenames:
            filepath = os.path.join(path, filename)

            yield image_utils.load_image(filepath)


def generate_batch_variable(image_generator, use_cuda, batch_size):
    batch = []

    for _ in range(batch_size):
        image = next(image_generator)
        image = image_utils.normalize(image)
        batch.append(image)

    batch = np.stack(batch).astype(np.float32)
    batch = np.transpose(batch, (0, 3, 1, 2))  # (batch, y, x, channel) -> (batch, channel, y, x)
    batch = torch.from_numpy(batch)

    if use_cuda:
        batch = batch.cuda()

    batch = torch.autograd.Variable(batch, requires_grad=False)

    return batch


def optimize(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def float_to_string(var):
    return '{:.4f}'.format(var.data[0])


def create_debug_images(a, b, gen_a_to_b, gen_b_to_a, use_cuda):
    gen_a_to_b.eval()  # Evaluation mode.
    gen_b_to_a.eval()

    b_fake = gen_a_to_b(a)
    a_fake = gen_b_to_a(b)

    batches = [a, b, b_fake, a_fake]

    batches = [b[:1, :, :, :] for b in batches]  # Extract first image only
    if use_cuda:
        batches = [b.cpu() for b in batches]  # Move data from gpu to cpu
    batches = [b.data.numpy() for b in batches]
    batches = [np.transpose(b, (0, 2, 3, 1)) for b in batches]  # (batch, channel, y, x) -> (batch, y, x, channel)
    batches = [image_utils.unnormalize(b[0, :, :, :]) for b in batches]

    gen_a_to_b.train()  # Train mode.
    gen_b_to_a.train()

    a, b, b_fake, a_fake = batches
    return a, b, b_fake, a_fake


if __name__ == '__main__':
    train()
