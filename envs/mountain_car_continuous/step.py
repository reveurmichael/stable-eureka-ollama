def step(self, action: np.ndarray):
    position = self.state[0]
    velocity = self.state[1]
    force = min(max(action[0], self.min_action), self.max_action)

    velocity += force * self.power - 0.0025 * math.cos(3 * position)
    if velocity > self.max_speed:
        velocity = self.max_speed
    if velocity < -self.max_speed:
        velocity = -self.max_speed
    position += velocity
    if position > self.max_position:
        position = self.max_position
    if position < self.min_position:
        position = self.min_position
    if position == self.min_position and velocity < 0:
        velocity = 0

    # Convert a possible numpy bool to a Python bool.
    terminated = bool(
        position >= self.goal_position and velocity >= self.goal_velocity
    )

    reward, individual_reward = self.compute_reward(position, velocity, action, terminated)

    fitness_score = self.compute_fitness_score(position, velocity, action, terminated)

    self.state = np.array([position, velocity], dtype=np.float32)

    if self.render_mode == "human":
        self.render()

    individual_reward.update({'fitness_score': fitness_score})

    return self.state, reward, terminated, False, individual_reward