from PIL import Image

from cfg import (
    logger,
    DEFAULT_IMAGE_SIZE
)


def get_image_matrix(image_dir):
    try:
        image_grayscale = Image.open(image_dir).convert('L')
        #resize image
        image_grayscale = image_grayscale.resize(DEFAULT_IMAGE_SIZE, Image.ANTIALIAS)

        image_np = np.array(image_grayscale)
        img_list = []
        for line in image_np:
            for value in line:
                img_list.append(value)
        return img_list
    except Exception as e:
        logger.error(e)
        return None
