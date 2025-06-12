from PyHJ.data import Batch


def find_a(obs, policy):
    assert isinstance(obs, dict), "Observation must be a dictionary."
    tmp_obs = {k: v.reshape((1,) + v.shape) for k, v in obs.items()}
    tmp_batch = Batch(obs=tmp_obs, info=Batch())
    tmp = policy(tmp_batch, model="actor_old").act
    act = policy.map_action(tmp).cpu().detach().numpy().flatten()
    return act


def evaluate_V(obs, policy, critic):
    assert isinstance(obs, dict), "Observation must be a dictionary."
    tmp_obs = {k: v.reshape((1,) + v.shape) for k, v in obs.items()}
    tmp_batch = Batch(obs=tmp_obs, info=Batch())
    tmp = critic(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()
