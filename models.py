
import torch.nn as nn
import torch.nn.functional as F


NUM_COLOR_CHANNELS = 3
NUM_GEN_FEATURES = 32
GEN_KERNEL_SIZE = 3

NUM_DISCR_FEATURES = 64
DISCR_KERNEL_SIZE = 4
DISCR_PADDING = 1

NUM_RESNET_BLOCKS = 6

RECONSTRUCTED_LOSS_WEIGHT = 10

LEAKY_RELU_NEGATIVE_SLOPE = 0.2


class GeneratorNet(nn.Module):
    def __init__(self):
        super().__init__()

        model = []

        # Encode:
        model += [
            nn.ReflectionPad2d(3),
            Conv2dWithBatchNorm(in_channels=NUM_COLOR_CHANNELS, out_channels=NUM_GEN_FEATURES,
                                kernel_size=7, stride=1, relu=True),
            Conv2dWithBatchNorm(in_channels=NUM_GEN_FEATURES, out_channels=NUM_GEN_FEATURES * 2,
                                kernel_size=GEN_KERNEL_SIZE, stride=2, padding=1, relu=True),
            Conv2dWithBatchNorm(in_channels=NUM_GEN_FEATURES * 2, out_channels=NUM_GEN_FEATURES * 4,
                                kernel_size=GEN_KERNEL_SIZE, stride=2, padding=1, relu=True),
        ]

        # Transform:
        model += [ResnetBlock(NUM_GEN_FEATURES * 4) for _ in range(NUM_RESNET_BLOCKS)]

        # Decode:
        model += [
            # Using nn.Upsample instead of nn.ConvTranspose2d, since nn.Upsample prevents
            # checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv2dWithBatchNorm(in_channels=NUM_GEN_FEATURES * 4, out_channels=NUM_GEN_FEATURES * 2,
                                kernel_size=GEN_KERNEL_SIZE, stride=1, padding=1, relu=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv2dWithBatchNorm(in_channels=NUM_GEN_FEATURES * 2, out_channels=NUM_GEN_FEATURES,
                                kernel_size=GEN_KERNEL_SIZE, stride=1, padding=1, relu=True),

            nn.ReflectionPad2d(3),

            # Don't use batch norm in last layer:
            nn.Conv2d(in_channels=NUM_GEN_FEATURES, out_channels=NUM_COLOR_CHANNELS, kernel_size=7),
            nn.Tanh(),
        ]

        # Decode using nn.ConvTranspose2d instead of nn.Upsample:
        # # Decode:
        # model += [
        #     nn.ConvTranspose2d(in_channels=NUM_GEN_FEATURES * 4, out_channels=NUM_GEN_FEATURES * 2,
        #                        kernel_size=GEN_KERNEL_SIZE, stride=2, padding=1, output_padding=1, bias=False),
        #     nn.BatchNorm2d(NUM_GEN_FEATURES * 2, affine=False),
        #     nn.LeakyReLU(negative_slope=LEAKY_RELU_NEGATIVE_SLOPE, inplace=True),
        #
        #     nn.ConvTranspose2d(in_channels=NUM_GEN_FEATURES * 2, out_channels=NUM_GEN_FEATURES,
        #                        kernel_size=GEN_KERNEL_SIZE, stride=2, padding=1, output_padding=1, bias=False),
        #     nn.BatchNorm2d(NUM_GEN_FEATURES, affine=False),
        #     nn.LeakyReLU(negative_slope=LEAKY_RELU_NEGATIVE_SLOPE, inplace=True),
        #
        #     nn.ReflectionPad2d(3),
        #
        #     # Don't use batch norm in last layer:
        #     nn.Conv2d(in_channels=NUM_GEN_FEATURES, out_channels=NUM_COLOR_CHANNELS, kernel_size=7),
        #     nn.Tanh(),
        # ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        model = [
            nn.ReflectionPad2d(1),
            Conv2dWithBatchNorm(in_channels=num_features, out_channels=num_features,
                                kernel_size=GEN_KERNEL_SIZE, relu=True),
            nn.ReflectionPad2d(1),
            Conv2dWithBatchNorm(in_channels=num_features, out_channels=num_features,
                                kernel_size=GEN_KERNEL_SIZE, relu=False),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        r = self.model(x)
        assert x.size() == r.size()

        return F.leaky_relu(x + r, negative_slope=LEAKY_RELU_NEGATIVE_SLOPE)


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()

        model = [
            Conv2dWithBatchNorm(in_channels=NUM_COLOR_CHANNELS, out_channels=NUM_DISCR_FEATURES,
                                kernel_size=DISCR_KERNEL_SIZE, stride=2, padding=DISCR_PADDING, relu=True),
            Conv2dWithBatchNorm(in_channels=NUM_DISCR_FEATURES, out_channels=NUM_DISCR_FEATURES * 2,
                                kernel_size=DISCR_KERNEL_SIZE, stride=2, padding=DISCR_PADDING, relu=True),
            Conv2dWithBatchNorm(in_channels=NUM_DISCR_FEATURES * 2, out_channels=NUM_DISCR_FEATURES * 4,
                                kernel_size=DISCR_KERNEL_SIZE, stride=2, padding=DISCR_PADDING, relu=True),
            Conv2dWithBatchNorm(in_channels=NUM_DISCR_FEATURES * 4, out_channels=NUM_DISCR_FEATURES * 8,
                                kernel_size=DISCR_KERNEL_SIZE, stride=1, padding=DISCR_PADDING, relu=True),

            # Don't use batch norm in last layer:
            nn.Conv2d(in_channels=NUM_DISCR_FEATURES * 8, out_channels=1,
                      kernel_size=DISCR_KERNEL_SIZE, stride=1, padding=DISCR_PADDING)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Conv2dWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu, stride=1, padding=0):
        super().__init__()

        affine = not relu  # Affine is redundant when using ReLU.

        model = [
            # Bias is redundant when using batch norm:
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        ]

        if relu:
            model += [nn.LeakyReLU(negative_slope=LEAKY_RELU_NEGATIVE_SLOPE, inplace=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def compute_generators_loss(gen_a_to_b, gen_b_to_a, discr_a, discr_b, a, b, cycle_criterion, discr_criterion,
                            one_array, a_fake_image_pool, b_fake_image_pool):
    b_fake = gen_a_to_b(a)
    a_fake = gen_b_to_a(b)

    a_fake_image_pool.put(a_fake)
    b_fake_image_pool.put(b_fake)

    a_reconstructed = gen_b_to_a(b_fake)
    a_reconstructed_loss = cycle_criterion(a_reconstructed, a)

    b_reconstructed = gen_a_to_b(a_fake)
    b_reconstructed_loss = cycle_criterion(b_reconstructed, b)

    # If discr_b_result == 1, then the discriminator thinks that this is a real image.
    discr_b_result = discr_b(b_fake)
    discr_b_fooled_loss = discr_criterion(discr_b_result, one_array)

    discr_a_result = discr_a(a_fake)
    discr_a_fooled_loss = discr_criterion(discr_a_result, one_array)

    return (a_reconstructed_loss + b_reconstructed_loss) * RECONSTRUCTED_LOSS_WEIGHT \
           + (discr_b_fooled_loss + discr_a_fooled_loss)


def compute_discr_loss(discr, img, fake_image_pool, criterion, zero_array, one_array):
    loss_real = criterion(discr(img), one_array)

    img_fake = fake_image_pool.get()
    loss_fake = criterion(discr(img_fake), zero_array)

    return (loss_real + loss_fake) / 2

