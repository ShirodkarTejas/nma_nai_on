#!/usr/bin/env python3
"""
Vectorized Environment Wrapper
Enables parallel execution of multiple environment instances using multiprocessing.
"""

import numpy as np
import multiprocessing as mp
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'get_attr':
                target = env
                for part in data.split('.'):
                    # Handle indices or keys if needed, but here we just need attributes
                    target = getattr(target, part)
                remote.send(target)
            elif cmd == 'set_attr':
                path, value = data
                parts = path.split('.')
                target = env
                for part in parts[:-1]:
                    target = getattr(target, part)
                setattr(target, parts[-1], value)
                remote.send(True)
            else:
                raise NotImplementedError(f"Command {cmd} not implemented")
    except EOFError:
        pass
    finally:
        env.close()

class SubprocVecEnv:
    """
    VecEnv that runs multiple environments in parallel in separate processes.
    """
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, env_fn))
                  for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.num_envs = num_envs

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_attr(self, attr_name):
        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]

    def set_attr(self, attr_name, value):
        for remote in self.remotes:
            remote.send(('set_attr', (attr_name, value)))
        return [remote.recv() for remote in self.remotes]
