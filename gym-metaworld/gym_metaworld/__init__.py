from gymnasium.envs.registration import register

register(
    id="gym_metaworld/metaworld-v0",
    entry_point="gym_metaworld.env:MetaWorldVitaEnv",
    nondeterministic=True,
)
