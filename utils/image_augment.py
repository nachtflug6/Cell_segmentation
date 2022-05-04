import torch as th
import torchvision.transforms as T


def uniform_slicing_img_stack(img_stack, output_shape, step_shape):

    img_t = th.unsqueeze(img_stack, 0)
    i_length = output_shape[0]
    j_length = output_shape[1]
    i_step = step_shape[0]
    j_step = step_shape[1]

    img_patches = img_stack.data.unfold(2, i_length, i_step).unfold(3, j_length, j_step)
    img_patches = th.flatten(img_patches, start_dim=2, end_dim=3)
    img_patches = th.swapaxes(img_patches, 1, 2)

    return img_patches


