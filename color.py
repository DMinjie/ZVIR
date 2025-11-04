import numpy as np
from PIL import Image

def gray_to_rgb_from_reference(gray_image, reference_rgb_image):

    gray_np = np.array(gray_image).astype(np.float32)
    ref_np = np.array(reference_rgb_image).astype(np.float32)

    Cb_ref = 128 + (-0.168736 * ref_np[:, :, 0] -
                    0.331264 * ref_np[:, :, 1] +
                    0.5 * ref_np[:, :, 2])
    Cr_ref = 128 + (0.5 * ref_np[:, :, 0] -
                    0.418688 * ref_np[:, :, 1] -
                    0.081312 * ref_np[:, :, 2])


    Y = gray_np
    Cb = Cb_ref
    Cr = Cr_ref


    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)


    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    rgb_np = np.stack([R, G, B], axis=-1)
    return Image.fromarray(rgb_np, mode='RGB')



