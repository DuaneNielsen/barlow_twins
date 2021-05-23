import datamodule

import pytest
import buffer_h5 as b5
import os
import numpy as np
from env.debug import DummyEnv

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
    os.remove('test.h5')


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
    state_col = b5.Column('state', (1, ), np.uint8, compression='gzip')
    raw_col = b5.Column('raw', (240, 160, 3), np.uint8, compression='gzip')
    action_col = b5.Column('action', dtype=np.int64, chunk_size=100000)
    b.create(filename, state_col=state_col, action_col=action_col, raw_col=raw_col)

    def policy(state):
        return state

    for step, s, a, s_p, r, d, i, m in b.step(env, policy, capture_raw=True):
        if step + 1 == len(t1) -1 + len(t2) - 1 + len(t3) - 1 + len(t4)-1 + len(t5) - 1:
            break

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