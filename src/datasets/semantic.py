import torch.nn as nn


def get_random_reflections(original_ds, batches, batch_size, original_img_size, output_img_size, normalized=False, ):

    new_ds = []
    num_examples = len(original_ds)
    img, seg = original_ds[0]
    img_shape = img.shape
    seg_shape = seg.shape

    imgs = th.zeros((num_examples, *img_shape))
    segs = th.zeros((num_examples, *seg_shape))

    for j, data in enumerate(original_ds, 0):
        img, seg = data
        imgs[j] = img
        segs[j] = seg

    if normalized:
        var, mean = th.var_mean(imgs)
        imgs = (imgs - mean) / var

    for i in range(batches):
        img_tensor = None
        seg_tensor = None

        for j in range(batch_size):
            random_index = np.random.randint(0, num_examples)
            img = imgs[random_index]
            segmentation = segs[random_index]
            new_img, new_segmentation = random_reflection(img, segmentation, original_img_size, output_img_size)

            if isinstance(img_tensor, type(None)):
                img_tensor = th.zeros((batch_size, *new_img.shape), dtype=th.float32)
                seg_tensor = th.zeros((batch_size, *new_segmentation.shape), dtype=th.float32)

            img_tensor[j] = new_img
            seg_tensor[j] = new_segmentation

        for j in range(batch_size):
            new_ds.append((img_tensor[j], seg_tensor[j]))

    return new_ds


def random_reflection(input_img_tensor, input_target_tensor, img_size, output_size):

    height, width = img_size
    transform = T.Resize(output_size)

    window_size = input_img_tensor.shape[-1]
    half_window_size = window_size // 2

    input_img_padded = nn.ReflectionPad2d(half_window_size)(input_img_tensor)
    input_target_padded = nn.ReflectionPad2d(half_window_size)(input_target_tensor)

    i_value = np.random.randint(0, window_size)
    j_value = np.random.randint(0, window_size)

    output_img_tensor = input_img_padded[:, i_value:(i_value+height), j_value:(j_value+width)]
    output_target_tensor = input_target_padded[:, i_value:(i_value+height), j_value:(j_value+width)]

    output_img_tensor = transform(output_img_tensor)
    output_target_tensor = transform(output_target_tensor)

    output_target_tensor = th.where(output_target_tensor < 0.5, 0.0, 1.0)
    sum_output_target_tensor = th.sum(output_target_tensor, dim=0)

    return output_img_tensor, output_target_tensor


def get_semantic_ds(path, classes: list, instances=False):
    ds = []

    images_path = os.path.join(path, 'images')
    if instances:
        labels_path = os.path.join(path, 'instance_labels')
    else:
        labels_path = os.path.join(path, 'labels')

    image_list = load_images_to_list_of_pil(images_path)
    label_list = load_images_to_list_of_pil(labels_path)

    num_images = len(image_list)

    for i in range(num_images):
        ds.append(tuple((transform_from_16bit_pil(image_list[i]), transform_semantic_tensor(label_list[i], classes))))

    return ds


def transform_semantic_tensor(semantic_pil_image, classes: list):

    img_arr = np.array(semantic_pil_image)
    height, width = img_arr.shape
    depth = len(classes)
    output_tensor = np.zeros((depth, height, width), dtype=float)

    for i in range(depth):
        current_class = classes[i]
        output_tensor[i] = np.where(img_arr == current_class, 1.0, 0.0)

    return th.from_numpy(output_tensor)
