import numpy as np

def add_noise(x, noise_level=0.2, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-noise_level, noise_level, size=x.shape)
    return x + noise

def input_stability(x, model, explainer, noise_level=0.2, eps=1e-6, seed=None):
    s = explainer(model, x)

    x_prime = add_noise(x, noise_level=noise_level, seed=seed)

    s_prime = explainer(model, x_prime)

    num = np.linalg.norm((s - s_prime).ravel())
    den = np.linalg.norm((x - x_prime).ravel()) + eps

    return float(num / den)