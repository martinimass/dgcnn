import os
import glob
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--block_size", type=float, default=1.0, help="Block Size, in meters (float)")
parser.add_argument("--stride", type=float, default=0.5, help="Stride, in meters (float)")
parser.add_argument("--skip_phase1", help="skip phase1", action="store_false", default=True)
args = parser.parse_args()

phase1 = args.skip_phase1
block_size = args.block_size
stride = args.stride

data_path = "data/"
areas_path = data_path+"dataset/"    #qui ci sono le cartelle Area1_...
meta_path = data_path+"meta/"   #dove si trova la cartella meta

######################################################################################################################
# FASE 1...
#
#  collect_indoor3d_data.py
#  serve:
#  - meta/anno_paths.txt
#  - data/dataset/Area...
######################################################################################################################


g_classes = [x.rstrip() for x in open(os.path.join(meta_path+'class_names.txt'))]
g_class2label = {cls: i for i,cls in enumerate(g_classes)}

def collect_point_label(anno_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
      We aggregated all the points from each instance in the room.

    Args:
      anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
      out_filename: path to save collected points and labels (each line is XYZRGBL)
      file_format: txt or numpy, determines what file format to save.
    Returns:
      None
    Note:
      the points are shifted before save, the most negative point is now at origin.
    """
    points_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        print(" - "+cls)
        if cls not in g_classes:  # note: in some room there is 'staris' class..
            cls = 'clutter'
        try:
            points = np.loadtxt(f)
            labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
            points_list.append(np.concatenate([points, labels], 1))  # Nx7
        except:
            print(" - - error...")

    data_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % \
                       (data_label[i, 0], data_label[i, 1], data_label[i, 2],
                        data_label[i, 3], data_label[i, 4], data_label[i, 5],
                        data_label[i, 6]))
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
              (file_format))
        exit()


anno_paths = [line.rstrip() for line in open(os.path.join(meta_path+'anno_paths.txt'))]
anno_paths = [os.path.join(areas_path, p) for p in anno_paths]

indoor3d_data_dir = os.path.join(data_path+'stanford_indoor3d')
if not os.path.exists(indoor3d_data_dir):
    os.mkdir(indoor3d_data_dir)

if phase1:
    print("Phase1...")
    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for anno_path in anno_paths:
        print(anno_path)
        try:
            elements = anno_path.split('/')
            out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
            collect_point_label(anno_path, os.path.join(indoor3d_data_dir, out_filename), 'numpy')
        except:
            print(anno_path, 'ERROR!!')

    ######################################################################################################################
    # creo file in meta/
    ######################################################################################################################
    print("Create some files in meta/")

    files_npy = glob.glob(indoor3d_data_dir + "/*.npy")
    files_npy_basename = []
    areas = {}
    for f in files_npy:
        basename = os.path.basename(f)
        files_npy_basename.append(basename)
        split = basename.split("_")
        id = int(split[1])
        if id not in areas:
            areas[id] = []
        areas[id].append(indoor3d_data_dir + "/" + basename)

    # all_data_label.txt
    with open(meta_path + "all_data_label.txt", "w") as fw:
        for f in files_npy_basename[:-1]:
            fw.write(f + "\n")
        fw.write(files_npy_basename[-1])

    # ogni area1_data_label.txt
    for id in areas:
        with open(meta_path + "area{}_data_label.txt".format(id), "w") as fw:
            if len(areas[id]) > 1:
                for f in areas[id][:-1]:
                    fw.write(f + "\n")
            fw.write(areas[id][-1])

else:
    print("Phase1 skipped!")



######################################################################################################################
# FASE 2...
# gen_indoor3d_h5.py
######################################################################################################################

NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
filelist = os.path.join(meta_path+'all_data_label.txt')
data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for line in open(filelist)]
output_dir = os.path.join(data_path, 'indoor3d_sem_seg_hdf5_data')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')


print("Phase2...")
print("- block_size:",block_size)
print("- stride:", stride)
fout_room = open(output_room_filelist, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save


######################################################################################################################
# FUNZIONI...
######################################################################################################################

def sample_data(data, num_sample):
  """ data is in N x ...
    we want to keep num_samplexC of them.
    if N > num_sample, we will randomly keep num_sample of them.
    if N < num_sample, we will randomly duplicate samples.
  """
  N = data.shape[0]
  if (N == num_sample):
    return data, range(N)
  elif (N > num_sample):
    sample = np.random.choice(N, num_sample)
    return data[sample, ...], sample
  else:
    sample = np.random.choice(N, num_sample-N)
    dup_data = data[sample, ...]
    return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)

def sample_data_label(data, label, num_sample):
  new_data, sample_indices = sample_data(data, num_sample)
  new_label = label[sample_indices]
  return new_data, new_label

def room2blocks(data, label, num_point, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1):
    """ Prepare block training data.
    Args:
      data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
        assumes the data is shifted (min point is origin) and aligned
        (aligned with XYZ axis)
      label: N size uint8 numpy array from 0-12
      num_point: int, how many points to sample in each block
      block_size: float, physical size of the block in meters
      stride: float, stride for block sweeping
      random_sample: bool, if True, we will randomly sample blocks in the room
      sample_num: int, if random sample, how many blocks to sample
        [default: room area]
      sample_aug: if random sample, how much aug
    Returns:
      block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
      block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * stride)
                ybeg_list.append(j * stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < 100:  # discard block if there are less than 100 pts.
            continue

        block_data = data[cond, :]
        block_label = label[cond]

        # randomly subsample data
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)
        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)

def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
      for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:, 0:6]
    data[:, 3:6] /= 255.0
    label = data_label[:, -1].astype(np.uint8)
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    data_batch, label_batch = room2blocks(data, label, num_point, block_size, stride,
                                          random_sample, sample_num, sample_aug)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2] / max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx + block_size / 2)
        data_batch[b, :, 1] -= (miny + block_size / 2)
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch

def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):
  if data_label_filename[-3:] == 'txt':
    data_label = np.loadtxt(data_label_filename)
  elif data_label_filename[-3:] == 'npy':
    data_label = np.load(data_label_filename)
  else:
    print('Unknown file type! exiting.')
    exit()
  return room2blocks_plus_normalized(data_label, num_point, block_size, stride, random_sample, sample_num, sample_aug)

# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


sample_cnt = 0
for i, data_label_filename in enumerate(data_label_files):
    print(data_label_filename)
    #data, label = room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1.0, stride=0.5, random_sample=False, sample_num=None)
    data, label = room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=block_size, stride=stride, random_sample=False, sample_num=None)
    print('{0}, {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    insert_batch(data, label, i == len(data_label_files)-1)

fout_room.close()
print("Total samples: {0}".format(sample_cnt))

#all_files.txt in output_dir
with open(output_dir+"/all_files.txt","w") as fw:
    files_h5 = glob.glob(output_dir+"/*.h5")
    for f in files_h5[:-1]:
        fw.write(output_dir+"/"+os.path.basename(f) + "\n")
    fw.write(os.path.basename(files_h5[-1]))

print("Done!")
