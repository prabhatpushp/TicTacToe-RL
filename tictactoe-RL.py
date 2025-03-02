import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.player_symbol = 'X'
        self.agent_symbol = 'O'

    def reset(self):
        self.board = [' '] * 9

    def is_winner(self, symbol):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for combo in winning_combinations:
            if all(self.board[i] == symbol for i in combo):
                return True
        return False

    def is_board_full(self):
        return ' ' not in self.board

    def is_game_over(self):
        return self.is_winner(self.player_symbol) or self.is_winner(self.agent_symbol) or self.is_board_full()

    def get_available_actions(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def print_board(self):
        for i in range(0, 9, 3):
            print(" | ".join(self.board[i:i+3]))
            if i < 6:
                print("---------")

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=1.0):
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor for future rewards
        self.q_table = {}       # Q-values table

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)  # Exploration
        else:
            q_values = [self.get_q_value(state, action) for action in available_actions]
            max_q = max(q_values)
            return random.choice([action for action, q_value in zip(available_actions, q_values) if q_value == max_q])

    def update_q_value(self, state, action, reward, next_state):
        max_next_q = max([self.get_q_value(next_state, next_action) for next_action in range(9)])
        current_q = self.get_q_value(state, action)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        self.q_table[(state, action)] = new_q

def play_game(agent, env, is_human=False):
    state = tuple(env.board)
    total_reward = 0

    while not env.is_game_over():
        if is_human:
            # Human player's turn
            env.print_board()
            try:
                print("Your move (1-9): ")
                player_action = int(input()) - 1
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if player_action not in env.get_available_actions():
                print("Invalid move. Try again.")
                continue
        else:
            # Agent's turn
            player_action = agent.choose_action(state, env.get_available_actions())

        env.board[player_action] = env.player_symbol
        state = tuple(env.board)
        reward = 0

        if env.is_winner(env.player_symbol):
            reward = 1
        elif env.is_winner(env.agent_symbol):
            reward = -1

        if not is_human:
            agent.update_q_value(state, player_action, reward, tuple(env.board))

        total_reward += reward

        if env.is_board_full() and not env.is_game_over():
            print("It's a tie!")
            break

        # Switch player
        env.player_symbol, env.agent_symbol = env.agent_symbol, env.player_symbol

    env.print_board()
    return total_reward



if __name__ == "__main__":
    env = TicTacToe()
    agent = QLearningAgent()

    num_training_episodes = 5000
    num_testing_episodes = 100

    # Training
    for episode in range(1, num_training_episodes + 1):
        env.reset()
        total_reward = play_game(agent, env)
        print(f"Training Episode {episode}/{num_training_episodes}, Total Reward: {total_reward}")

    print("Training complete. Now you can play against the trained agent.")

    env.reset();
    # Play against the trained agent
    play_game(agent, env, is_human=True)
