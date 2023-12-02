import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import random
def plot_learning_curve(x,scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Moving average of previous 100 scores')
    plt.xlabel('Moving average of episode')
    plt.ylabel('Return')
    plt.savefig(figure_file)
    plt.close()

def plot_action_curve(action, figure_file):
    plt.figure(figsize=(8, 6))
    plt.plot(action, label='Action', color='black')
    plt.xlabel('Time')
    plt.ylabel('Action Value')
    plt.title('Action Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_file)
    plt.close()


def plot_reward_curve(reward, figure_file):
    plt.figure(figsize=(8, 6))
    plt.plot(reward, label='Reward', color='red')
    plt.xlabel('Time')
    plt.ylabel('Reward Value')
    plt.title('Reward Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_file)
    plt.close()



def get_filepath(industrynumber):
    path_list = []
    industry_path = f"./Data/Industry_{industrynumber}/*"
    firms_list = glob.glob(industry_path)

    for firm in firms_list:
        firm_path = f"{firm}/*"
        event_lists = glob.glob(firm_path)
        for event in event_lists:
            path_list.append(event)

    return path_list

def get_agent_path():
    path_list = []
    agent_path = f"./tmp/sac/*"
    firms_list = glob.glob(agent_path)

    for firm in firms_list:
        firm_path = f"{firm}/*"
        event_lists = glob.glob(firm_path)
        for event in event_lists:
            path_list.append(event)

    return path_list

def find_ticker(ticker):
    pass


def select_events(input_list:[]):
    random.seed(42)
    list_size = len(input_list)
    # half_size = int(list_size // 2)

    selected_numbers = random.sample(input_list, 5)
    remaining_numbers = [num for num in input_list if num not in selected_numbers]
    random_remaining_number = random.choice(remaining_numbers)

    return selected_numbers, random_remaining_number




