import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from craftax.craftax_classic.envs.craftax_symbolic_env import get_map_obs_shape
from craftax.craftax_env import make_craftax_env_from_name

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState


class Goal(NamedTuple):
    inv_index: int

