import gymnasium as gym


class HolePenaltyWrapper(gym.Wrapper):
    """
    Wrapper de sécurité pour FrozenLake.

    La récompense standard de FrozenLake donne +1 au but et 0 sinon.
    Ce wrapper ajoute une pénalité lorsque l'agent tombe dans un trou.

    Récompense utilisée :
        +1  si l'état suivant est G
        -C  si l'état suivant est H
         0  sinon
    """

    def __init__(self, env: gym.Env, hole_penalty: float = -1.0):
        super().__init__(env)
        self.hole_penalty = hole_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        desc = self.unwrapped.desc.astype(str)
        n_cols = desc.shape[1]
        row, col = divmod(int(obs), n_cols)
        tile = desc[row, col]

        if tile == "H":
            reward = self.hole_penalty
            info["fell_in_hole"] = True
        else:
            info["fell_in_hole"] = False

        return obs, reward, terminated, truncated, info