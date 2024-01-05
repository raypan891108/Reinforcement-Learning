import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    #learning_rate：學習率，是一個介於 0 到 1 之間的數值，用於控制每次更新 Q 值時新資訊的影響程度。
    #discount_factor：折扣因子，是一個介於 0 到 1 之間的數值，用於表示未來回饋的折扣程度。這個值越大，越重視未來的回饋。
    #exploration_prob：探索機率，是一個介於 0 到 1 之間的數值，表示在選擇動作時進行探索的機率。如果隨機數小於 exploration_prob，則進行探索，否則進行利用。
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Initialize Q-values to zeros
        self.q_values = np.zeros((num_states, num_actions))

    def select_action(self, state):
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.num_actions)  # Explore
        else:
            return np.argmax(self.q_values[state, :])  # Exploit

    def update_q_values(self, state, action, reward, next_state):
        # Q-value update using the Q-learning formula
        self.q_values[state, action] = (1 - self.learning_rate) * self.q_values[state, action] + \
                                       self.learning_rate * (reward + self.discount_factor * np.max(self.q_values[next_state, :]))

def run_q_learning():
    # Define the maze as a grid (0: empty, 1: obstacle, 2: goal)
    maze = np.array([
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 2, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    num_states = np.prod(maze.shape)
    num_actions = 4  # 4 possible actions: up, down, left, right

    agent = QLearningAgent(num_states, num_actions)

    num_episodes = 2000

    best_reward = float('-inf')
    worst_reward = float('inf')
    total_rewards = []

    # 新增一個列表來儲存每個回合的平均報酬
    episode_rewards = []

    # Track optimal path
    optimal_path = []

    for episode in range(num_episodes):

        
        state = 0  # Start from the top-left corner

        prev_row = 0
        prev_col = 0
        total_reward = 0
        episode_path = []  # Track actions for this episode
        has_hit_obstacle = False # 是否有走到 1

        while True:
            action = agent.select_action(state)
            episode_path.append(action)  # Record the action

            # Simulate the environment (update the state based on the action)
            next_state = take_action(state, action, maze)
            reward = get_reward(next_state, maze, prev_row, prev_col)
            prev_row, prev_col = np.unravel_index(state, maze.shape)
            
            if reward == -1:  # 走到 1
                has_hit_obstacle = True

            # Update Q-values
            agent.update_q_values(state, action, reward, next_state)

            total_reward += reward
            state = next_state

            if maze.flat[next_state] == 2 :  # Reached the goal
                if (len(optimal_path) > len(episode_path) and not has_hit_obstacle) or len(optimal_path) == 0:
                    optimal_path = episode_path  # Update optimal path
                    # Output the optimal path
                    print("Optimal Path:", optimal_path)
                    print("len:", len(optimal_path))
                    print("times:", episode)
                has_hit_obstacle = False
                break

                #Optimal Path: [1, 1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 1, 3, 3, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2]
                #len: 24
                #times: 589

        # 更新最佳、最差成績以及瓶頸
        if total_reward > best_reward :
            best_reward = total_reward
        if total_reward < worst_reward :
            worst_reward = total_reward
        # 將成績加入列表
        total_rewards.append(total_reward)

        
        # 計算平均成績
        average_reward = np.mean(total_rewards)
        # 將每個回合的平均報酬添加到列表中
        episode_rewards.append(average_reward)
    # 只選擇最後100次的平均值
    last_100_episode_rewards = episode_rewards[-100:]

    

    # 設置畫布，建立兩個子圖，一行兩列
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    # 繪製 Average Reward 的折線圖
    ax1.plot(range(1, num_episodes + 1), episode_rewards, label='Average Reward')
    ax1.set_title('Average Reward over Episodes')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward')
    # ax1.set_ylim(-0.4, 0)
    ax1.legend()
  

    # 繪製 Q-values 的熱度圖在第二個子圖
    im = ax2.imshow(agent.q_values.T, cmap='viridis', aspect='auto', origin='lower')
    ax2.set_title('Q-values Heatmap')
    ax2.set_xlabel('States')
    ax2.set_ylabel('Actions')
    ax2.legend()

    # 繪製第三張圖，表示最後100次的平均值
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(range(num_episodes - 99, num_episodes + 1), last_100_episode_rewards, label='Average Reward')
    ax3.set_title('Average Reward over Last 100 Episodes')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Average Reward')
    ax3.legend()
    plt.colorbar(im, ax=ax2, label='Q-value')
    plt.tight_layout()
    plt.show()

    print(f"Best Reward: {best_reward}")
    print(f"Worst Reward: {worst_reward}")
    print(f"Average Reward : {average_reward}")
    print(f"Q-values:\n{agent.q_values}")

    

def take_action(state, action, maze):
    # Simulate the environment and return the next state based on the action
    num_cols = maze.shape[1]
    row, col = divmod(state, num_cols)

    if action == 0 :  # Move up
        row = max(0, row - 1)
    elif action == 1:  # Move down
        row = min(maze.shape[0] - 1, row + 1)
    elif action == 2 :  # Move left
        col = max(0, col - 1)
    elif action == 3:  # Move right
        col = min(maze.shape[1] - 1, col + 1)

    return row * num_cols + col

def get_reward(state, maze, prev_row, prev_col):
    # Get the current position (row, col) from the flattened state
    row, col = np.unravel_index(state, maze.shape)

    # Check if the agent is at a barrier
    if maze.flat[state] == 1:
        return -1
    # Check if the agent has moved to a new position
    elif row != prev_row or col != prev_col:
        return 0
    # If the agent is still at the same position after the action, apply a penalty
    else:
        return -0.1


if __name__ == "__main__":
    run_q_learning()
