import os
import logging
import collections

import matplotlib.pyplot as plt
import numpy as np

from . import networks
from ..envs_old.gridworld import gridworld_envs
from ..tools import flag_tools

class LapReprConfig(flag_tools.ConfigBase):

    def _set_default_flags(self):
        flags = self._flags
        flags.device = None
        flags.env_name = None
        flags.env_family = None
        # agent
        flags.d = 20
        flags.n_samples = 10000
        flags.batch_size = 128
        flags.discount = 0.9
        flags.w_neg = 1.0
        flags.c_neg = 1.0
        flags.reg_neg = 0.0
        flags.replay_buffer_size = 100000
        # train
        flags.log_dir = './log/generalized'
        flags.total_train_steps = 50000
        flags.print_freq = 1000
        flags.save_freq = 10000

    def _build(self):
        # self._build_env()
        self._build_args()

    def _obs_prepro(self, obs):
        return obs

    def _env_factory(self):
        raise NotImplementedError

    def _build_env(self):
        dummy_env = self._env_factory()
        dummy_time_step = dummy_env.reset()
        self._action_spec = dummy_env.action_spec
        self._obs_shape = list(self._obs_prepro(
            dummy_time_step.observation).shape)

    def _build_args(self):
        args = flag_tools.Flags()
        args.device = self._flags.device
        # env args
        # args.action_spec = self._action_spec
        # args.obs_shape = self._obs_shape
        args.obs_prepro = self._obs_prepro
        args.env_factory = self._env_factory
        # learner args
        args.n_samples = self._flags.n_samples
        args.batch_size = self._flags.batch_size
        args.discount = self._flags.discount
        args.w_neg = self._flags.w_neg
        args.c_neg = self._flags.c_neg
        args.reg_neg = self._flags.reg_neg
        args.replay_buffer_size = self._flags.replay_buffer_size
        # training args
        args.log_dir = self._flags.log_dir
        args.total_train_steps = self._flags.total_train_steps
        args.print_freq = self._flags.print_freq
        args.save_freq = self._flags.save_freq
        self._args = args

    @property
    def args(self):
        return vars(self._args)

    @property
    def args_as_flags(self):
        return self._args


class Config(LapReprConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        flags.d = 10
        flags.n_samples = 100_000
        flags.batch_size = 32
        flags.discount = 0.9
        flags.w_neg = 5.0
        flags.c_neg = 1.0
        flags.reg_neg = 0.0
        flags.replay_buffer_size = 100_000
        # train
        flags.log_dir = '/tmp/rl_laprepr/log'
        flags.total_train_steps = 200_000
        flags.print_freq = 1000
        flags.save_freq = 10000

        flags.max_distance = 9

    # def _obs_prepro(self, obs):
    #     if 'cube' in self._flags.env_name:
    #         return obs
    #     else:
    #         return obs.agent.position

    def _env_factory(self):
        env =  gridworld_envs.make(self._flags.env_name)
        return env
