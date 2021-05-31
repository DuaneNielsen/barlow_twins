import datamodule

import pytest
import buffer_h5 as b5
import os
import numpy as np
from env.debug import DummyEnv
from datamodule import H5NextStateReward, write_balanced_splits, H5ImageLabelDataset, PolicyActionLabels
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def identity(x):
    return x


def transition_equal(t1, t2):
    for i, field in enumerate(t1):
        if isinstance(field, dict):
            pass
        elif isinstance(field, np.ndarray):
            assert np.allclose(t1[i], t2[i])
        else:
            assert t1[i] == t2[i]


@pytest.fixture
def filename():
    if os.path.exists('test.h5'):
        os.remove('test.h5')
    yield 'test.h5'
    if os.path.exists('test.h5'):
        os.remove('test.h5')


@pytest.fixture
def dest_filename():
    if os.path.exists('load.h5'):
        os.remove('load.h5')
    yield 'load.h5'
    if os.path.exists('load.h5'):
        os.remove('load.h5')


shape = (210, 160, 3)
dtype = np.uint8

s1 = np.random.randint(shape, dtype=dtype)
a1 = np.random.randint(1, dtype=np.uint8)
r1 = 0.0
d1 = False

s2 = np.random.randint(shape, dtype=dtype)
a2 = np.random.randint(1, dtype=np.uint8)
r2 = 0.0
d2 = False

t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([1]), 0.0, False, {}),
      (np.array([2]), 0.0, False, {}), (np.array([3]), 0.0, False, {}), (np.array([4]), 1.0, True, {})]
t2 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([1]), 0.0, False, {}),
      (np.array([2]), 0.0, False, {}), (np.array([3]), 1.0, True, {})]
t3 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([1]), 0.0, False, {}),
      (np.array([2]), 1.0, True, {})]
t4 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([1]), 0.0, False, {}),
      (np.array([2]), 0.0, True, {})]
t5 = [(np.array([0]), 0.0, False, {}), (np.array([2]), 1.0, True, {})]


def populated_buffer(filename):
    traj = [t1, t2, t3, t4, t5]

    env = DummyEnv(traj)
    b = b5.Buffer()
    state_col = b5.NumpyColumn('state', (1,), np.uint8, compression='gzip')
    raw_col = b5.NumpyColumn('raw', (240, 160, 3), np.uint8, compression='gzip')
    grad_col = b5.NumpyColumn('grad', (240, 160, 3), np.uint8, compression='gzip')
    action_col = b5.NumpyColumn('action', dtype=np.int64, chunk_size=100000)
    b.create(filename, state_col=state_col, action_col=action_col, raw_col=raw_col)

    def policy(state):
        return state

    for step, s, a, s_p, r, d, i, m in b.step(env, policy, capture_raw=True):
        if step + 1 == len(t1) -1 + len(t2) - 1 + len(t3) - 1 + len(t4)-1 + len(t5) - 1:
            break

    grad_col.create(b.replay)
    b.f['/replay/grad'].resize(b.steps, axis=0)
    return b


def test_r_distance(filename):
    b = populated_buffer(filename)
    b.close()

    b = datamodule.H5NextStateReward()
    b.transforms = identity
    b.load(filename)
    count = 0
    x = []
    assert len(b) == len(t1) + len(t2) + len(t3) + len(t4) + len(t5)
    assert len(b) == 21
    for item in range(len(b)):
        count += b[item][1]
        x.append(b[item][1])
    assert count == 17
    b.close()

    b = datamodule.H5NextStateReward()
    b.transforms = identity
    b.load(filename, reward_causality_distance=2)
    count = 0
    x = []
    for item in range(len(b)):
        count += b[item][1]
        x.append(b[item][1])
    assert count == 11
    b.close()

    b = datamodule.H5NextStateReward()
    b.transforms = identity
    b.load(filename, reward_causality_distance=0)
    count = 0
    x = []
    for item in range(len(b)):
        count += b[item][1]
        x.append(b[item][1])
    assert count == 4
    b.close()

    b = datamodule.H5NextStateReward()
    b.transforms = identity
    b.load(filename, reward_causality_distance=1)
    count = 0
    x = []
    for item in range(len(b)):
        count += b[item][1]
        x.append(b[item][1])
    assert count == 8
    b.close()


def test_write_new(filename, dest_filename):
    b = populated_buffer(filename)
    b.close()
    ds = H5NextStateReward()
    ds.transforms = identity
    ds.load(filename)
    class_dict = {'no_reward': 0, 'reward': 1}
    write_balanced_splits(ds, dest_filename, class_dict, 256, 10, 'gzip', 6)
    batch_ds = H5ImageLabelDataset(dest_filename, 'train', ToTensor(), batch_size=256)
    dl = DataLoader(batch_ds, num_workers=0, batch_size=None)
    for image, label in dl:
        assert image.shape[0] == 256
        assert image.shape[1] == 6
        assert image.shape[2] == 240
        assert image.shape[3] == 160
        assert image.shape[0] == 256


def test_policy_action_labels(filename, dest_filename):
    b = populated_buffer(filename)
    b.close()
    ds = PolicyActionLabels()
    ds.transforms = identity
    ds.load(filename)
    class_dict = dict(zip(ds.name_classes, range(len(ds.name_classes))))
    write_balanced_splits(ds, dest_filename, class_dict, 256, 10, 'gzip', 6)
    batch_ds = H5ImageLabelDataset(dest_filename, 'train', ToTensor(), batch_size=256)
    dl = DataLoader(batch_ds, num_workers=0, batch_size=None)
    for image, label in dl:
        assert image.shape[0] == 256
        assert image.shape[1] == 6
        assert image.shape[2] == 240
        assert image.shape[3] == 160
        assert image.shape[0] == 256
