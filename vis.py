import torch  # https://github.com/microsoft/onnxruntime/issues/11092#issuecomment-1386840174
import onnxruntime as ort

from env import Env

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 1,
    }),
    'CPUExecutionProvider',
]
policy = ort.InferenceSession("policy.onnx", providers=providers)
env = Env(device=torch.device("cpu"))

obs, _ = env.reset(seed=None)
for step in range(1000):
    action = policy.run(None, {"state": obs.numpy()})
    next_obs, reward, terminated, truncated, _ = env.step(action[0])

    if not terminated and not truncated:
        obs = next_obs
    else:
        obs, _ = env.reset()
