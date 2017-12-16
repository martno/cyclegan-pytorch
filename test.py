import click
from tqdm import tqdm
from os.path import join as join_path
import skimage.io
import torch
import os.path
import numpy as np
import natural.number

import models
import model_utils
import utils
import constants as c
import image_utils
from utils import Params, Timer, ensure_dir_exists


@click.command()
@click.option('--dataset_a', type=click.Path(exists=True), help='Path to dataset A')
@click.option('--dataset_b', type=click.Path(exists=True), help='Path to dataset B')
@click.option('--use_cuda/--no_cuda', default=False, show_default=True)
@click.option('--checkpoint_path', default='checkpoint', show_default=True, type=click.Path(exists=True, dir_okay=True), help='Checkpoint path')
@click.option('--test_save_path', default='test-output', show_default=True, help='Folder to save test images')
def test(**kwargs):
    params = Params(kwargs)

    print('Params:')
    params.pretty_print()
    print()

    use_cuda = params.use_cuda
    if use_cuda:
        assert torch.cuda.is_available()

    with Timer('Loading models'):
        gen_a_to_b, gen_b_to_a = load_models_for_evaluation(params.checkpoint_path)

    print('#weights in gen_a_to_b:', natural.number.number(model_utils.compute_num_weights(gen_a_to_b)))
    print('#weights in gen_b_to_a:', natural.number.number(model_utils.compute_num_weights(gen_b_to_a)))

    if use_cuda:
        gen_a_to_b.cuda()
        gen_b_to_a.cuda()

    a_to_b_save_path = join_path(params.test_save_path, c.A_TO_B_GEN_TEST_DIR)
    b_to_a_save_path = join_path(params.test_save_path, c.B_TO_A_GEN_TEST_DIR)

    ensure_dir_exists(a_to_b_save_path)
    ensure_dir_exists(b_to_a_save_path)

    filenames = utils.listdir(params.dataset_a, extensions=('.png', '.jpg'))
    for filename in tqdm(filenames, desc='A to B'):
        filepath = join_path(params.dataset_a, filename)

        a = image_utils.load_image(filepath)

        b_fake = generate_fake_image(image=a, generator_net=gen_a_to_b, use_cuda=use_cuda)

        root, ext = os.path.splitext(filename)
        a_filepath = join_path(a_to_b_save_path, '{}-a{}'.format(root, ext))
        skimage.io.imsave(a_filepath, a)

        a_to_b_filepath = join_path(a_to_b_save_path, '{}-a-to-b{}'.format(root, ext))
        skimage.io.imsave(a_to_b_filepath, b_fake)

    filenames = utils.listdir(params.dataset_b, extensions=('.png', '.jpg'))
    for filename in tqdm(filenames, desc='B to A'):
        filepath = join_path(params.dataset_b, filename)

        b = image_utils.load_image(filepath)

        a_fake = generate_fake_image(image=b, generator_net=gen_b_to_a, use_cuda=use_cuda)

        root, ext = os.path.splitext(filename)
        b_filepath = join_path(b_to_a_save_path, '{}-b{}'.format(root, ext))
        skimage.io.imsave(b_filepath, b)

        b_to_a_filepath = join_path(b_to_a_save_path, '{}-b-to-a{}'.format(root, ext))
        skimage.io.imsave(b_to_a_filepath, a_fake)


def load_models_for_evaluation(checkpoint_path):
    assert os.path.exists(checkpoint_path), checkpoint_path

    gen_a_to_b = models.GeneratorNet()
    gen_b_to_a = models.GeneratorNet()

    gen_a_to_b.load_state_dict(torch.load(join_path(checkpoint_path, c.A_TO_B_GEN_DIR)))
    gen_b_to_a.load_state_dict(torch.load(join_path(checkpoint_path, c.B_TO_A_GEN_DIR)))

    gen_a_to_b.eval()  # Evaluation mode.
    gen_b_to_a.eval()

    return gen_a_to_b, gen_b_to_a


def generate_fake_image(image, generator_net, use_cuda):
    image = image_utils.normalize(image)
    image = image[np.newaxis, :, :, :]

    image = np.transpose(image, (0, 3, 1, 2))  # (batch, y, x, channel) -> (batch, channel, y, x)
    image = torch.from_numpy(image)
    if use_cuda:
        image = image.cuda()
    image = torch.autograd.Variable(image, requires_grad=False)

    fake = generator_net(image)

    fake = fake[0, :, :, :]
    if use_cuda:
        fake = fake.cpu()
    fake = fake.data.numpy()
    fake = np.transpose(fake, (1, 2, 0))  # (channel, y, x) -> (y, x, channel)
    fake = image_utils.unnormalize(fake[:, :, :])

    return fake


if __name__ == '__main__':
    test()
