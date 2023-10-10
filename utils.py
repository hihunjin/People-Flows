import h5py
import torch
import shutil
import imageio

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_with_imageio(output_filename: str, img_array: list, fps: int = 30):
    writer = imageio.get_writer(output_filename, fps=fps)
    for img in img_array:
        writer.append_data(img)
    writer.close()

    print(f"saved at {output_filename}")
