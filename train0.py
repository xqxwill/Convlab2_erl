# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""
import sys, os
import time

import numpy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append("/home/x/P/ConvLab-2/")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import random
import torch
import copy
import json
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.env import Environment
from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.ppo import PPO
from argparse import ArgumentParser
from convlab2.policy.rlmodule import Memory, Transition
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from argparse import ArgumentParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)
try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

def test(env, policy, simsz, i):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    res = {}

    for episode in range(simsz):

        # goal_id = random.randint(0, len(all_goal_set) - 1)
        # goal = copy.deepcopy(all_goal_set[goal_id])
        # s = env.reset_new(goal)
        s = env.reset()
        done = False
        turn = 0

        while (not done):
            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)

            # interact with env
            next_s, r, done = env.step(a)
            cumulative_reward += r
            turn += 2

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # update per step
            s = next_s

            if done:
                if r>0:
                    successes += 1
                    print("test{} epoch {}: success, reward={}".format(i,episode,r))
                    print("goals:",env.usr.policy.policy.goal.domain_goals)
                else:
                    print("test{} epoch {}: fail, reward={}".format(i,episode,r))
                cumulative_turns += turn

    res['success_rate'] = float(successes) / simsz
    res['ave_reward'] = float(cumulative_reward) / simsz
    res['ave_turns'] = float(cumulative_turns) / simsz
    print("test epoch %s, success rate %s, ave reward %s, ave turns %s" % (
        i, res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res

def save_best_model(path, success_rate, model, best_epoch):
    print('saving best model...')
    filename = 'best_model_%s_epoch_%s_success.p' % (best_epoch, success_rate)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    checkpoint['state_dict'] = {k: v.cpu() for k, v in model.net.state_dict().items()}
    open(filepath, 'a').close()
    try:
        torch.save(checkpoint, filepath)
        print('saved best model in %s' % (filepath, ))
    except Exception:
        print('Error: Writing best model fails: %s' % (filepath, ))

def save_performance_records(path, records):
    print('saving performance epoch...')
    if not os.path.exists(path):
        os.makedirs(path)
    filename = 'performance.json'
    filepath = os.path.join(path, filename)

    try:
        json.dump(records, open(filepath, "w"))
        print('saved performance in %s' % (filepath, ))
    except Exception:
        print("Error: Writing model fails: %s" % (filepath, ))

def sampler(pid, queue, evt, env, policy, batchsz, epoch):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask)

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                if r>0:
                    print("train {}: success, reward={}".format(epoch,r))
                    # if len(env.usr.policy.policy.goal.domains)>1:
                    # print("goals:",env.usr.policy.policy.goal.domain_goals)
                else:
                    print("train {}: fail, reward={}".format(epoch,r))
                # print("GOAL-final", goal)
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


def sample(env, policy, batchsz, process_num, epoch):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
	:param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz, epoch)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_ = queue.get()
        buff0.append(buff_)  # merge current Memory into buff0
    evt.set()

    # now buff saves all the sampled data
    buff = buff0

    return buff.get_batch()


def update(env, policy, batchsz, epoch, process_num):
    # sample data asynchronously
    batch = sample(env, policy, batchsz, process_num, epoch)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    batchsz_real = s.size(0)
    print("-------------------update----------------------")
    time.sleep(2)
    policy.update(epoch, batchsz_real, s, a, r, mask)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="", help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=1024, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=2000, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=8, help="number of processes of trajactory sampling")
    parser.add_argument("--save_performance_dir", type=str, default="save")
    parser.add_argument('--torch_seed', dest='torch_seed', type=int, default=100, help='random seed for troch')
    args = parser.parse_args()

    torch.manual_seed(args.torch_seed)
    seed = 10
    numpy.random.seed(seed)
    random.seed(seed)
    # simple rule DST
    dst_sys = RuleDST()

    policy_sys = PPO(True)
    # policy_sys.load(args.load_path)

    # not use dst
    dst_usr = None
    # rule policy
    policy_usr = RulePolicy(character='usr')
    # assemble
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')

    # evaluator = MultiWozEvaluator()
    # env = Environment(None, simulator, None, dst_sys, evaluator)
    env = Environment(None, simulator, None, dst_sys, None)

    best_model = {}
    best_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'epoch': 0}
    best_model['model'] = copy.deepcopy(policy_sys)

    performance_records = {}
    performance_records['success_rate'] = {}
    performance_records['ave_turns'] = {}
    performance_records['ave_reward'] = {}

    for i in range(args.epoch):
        update(env, policy_sys, args.batchsz, i, args.process_num)
        if i == 0 or (i + 1) % 10 == 0:
            res = test(env, policy_sys, 50, i)
            performance_records['success_rate'][i] = res['success_rate']
            performance_records['ave_turns'][i] = res['ave_turns']
            performance_records['ave_reward'][i] = res['ave_reward']

            if res['success_rate'] > best_res['success_rate']:
                best_model['model'] = copy.deepcopy(policy_sys)
                best_res['success_rate'] = res['success_rate']
                best_res['ave_reward'] = res['ave_reward']
                best_res['ave_turns'] = res['ave_turns']
                best_res['epoch'] = i

            save_performance_records(args.save_performance_dir, performance_records)

    save_best_model(args.save_performance_dir, best_res['success_rate'], best_model['model'], best_res['epoch'])
    save_performance_records(args.save_performance_dir, performance_records)