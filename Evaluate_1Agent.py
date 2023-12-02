import pandas as pd
import numpy as np
from sac_torch import Agent
from utils import  plot_action_curve, plot_reward_curve
from ContinuousStockTradingEnv import ContinuousStockTradingEnv
'''
먼저 Train_1Agent.py 로 모델을 학습 시킨 후 리스크 성향을 측정하기 위한 코드 
'''

filename = 'StockTrade.png'
figure_file_ActionForTest = 'plots/Test/action/' + filename
figure_file_RewardForTest = 'plots/Test/reward/' + filename

test_df = pd.read_csv('./Data/Industry_1/TPR/2023-06-22 00_00_00.csv')
test_df.drop(['industry','event'],axis=1, inplace=True)
test_env = ContinuousStockTradingEnv(test_df)  # 테스트용 환경 생성
test_agent = Agent(input_dims=[14], env=test_env, n_actions=1, event_dir='')  # 테스트용 Agent 생성

# 학습된 모델 로드
test_agent.load_models()  # 학습된 모델 로드

# 테스트 데이터셋에서의 테스트 수행
test_observation = test_env.reset()
test_done = False
test_score = 0
test_action_history = []
test_reward_history = []

while not test_done:
    test_action = test_agent.choose_action(test_observation)
    test_observation_, test_reward, test_done, _ = test_env.step_(test_action)

    test_score += test_reward
    print(test_reward, 'type: ', type(test_reward))

    test_reward_history.append(test_reward)
    test_action_history.append(test_action[0])

    test_observation = test_observation_



metrics_df = pd.DataFrame({'Action': test_action_history, 'Reward': test_reward_history})
print(metrics_df)
# plot_action_curve(metrics_df['Action'], figure_file_ActionForTest)
# plot_reward_curve(metrics_df['Reward'], figure_file_RewardForTest)


