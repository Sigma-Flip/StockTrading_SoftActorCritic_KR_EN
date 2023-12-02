import pandas as pd
import numpy as np
from sac_torch import Agent
from utils import  plot_action_curve, plot_reward_curve,  get_agent_path
from ContinuousStockTradingEnv import ContinuousStockTradingEnv
import os
'''

* 먼저 Multi_N_Agent.py 로 모델을 학습시킨 후 리스크 성향을 측정하기 위한 코드 

'''
def evaluate(ticker,test_path):
    expected_return = None
    expected_action = None
    var = None

    try:
        reward_df = pd.DataFrame()
        action_df = pd.DataFrame()
        agent_path_list = get_agent_path()
        num_agent = 1
        for agent_path in agent_path_list:
            test_df = pd.read_csv(test_path)
            # date_list = test_df['Date'][:-1]
            date_list = list(range(1,30))
            test_df.drop(['industry', 'event'], axis=1, inplace=True)
            event_date = agent_path.split('/')[-1]
            event_firm = agent_path.split('/')[-2]

            event_dir = f'{event_firm}/{event_date}'
            os.makedirs(f'plots/{ticker}/action', exist_ok=True)
            os.makedirs(f'plots/{ticker}/reward', exist_ok=True)
            figure_file_ActionForTest = f'plots/{ticker}/action/agent{num_agent}_ActionGraph.png'
            figure_file_RewardForTest = f'plots/{ticker}/reward/agent{num_agent}_RewardGraph.png'


            test_env = ContinuousStockTradingEnv(test_df)  # 테스트용 환경 생성
            test_agent = Agent(input_dims=[14], env=test_env, n_actions=1, event_dir=event_dir)  # 테스트용 Agent 생성

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
                test_reward_history.append(test_reward)
                test_action_history.append(test_action[0])

                test_observation = test_observation_

            df_r = pd.DataFrame({'Reward': test_reward_history})
            df_a = pd.DataFrame({'Action': test_action_history})

            plot_action_curve(df_a['Action'], figure_file_ActionForTest)
            plot_reward_curve(df_r['Reward'], figure_file_RewardForTest)
            reward_df = pd.concat([reward_df, df_r], axis=1)
            action_df = pd.concat([action_df, df_a], axis=1)
            num_agent+=1

        # reward_df.to_csv(f'RiskEvaluation/{test_firm}_reward.csv', index=False)
        # action_df.to_csv(f'RiskEvaluation/{test_firm}_action.csv', index=False)

        reward_df.set_index(pd.Index(date_list), inplace=True)
        action_df.set_index(pd.Index(date_list), inplace=True)

        reward_df_mean = reward_df.mean(axis=1)
        action_df_mean = action_df.mean(axis=1)
        reward_df_var = reward_df.var(axis=1)
        result_df = pd.DataFrame({'Ave action':action_df_mean,'Ave reward':reward_df_mean,
                                  'Var reward':reward_df_var})

        result_df.to_csv(f'{ticker}_result.csv')


        reward_row = reward_df.loc[1]
        action_row = action_df.loc[1]

        expected_return = reward_row.mean()
        expected_action = action_row.mean()
        print('return: ',expected_return)
        print('eaction : ', expected_action)
        var = action_row.var()

    except FileNotFoundError as e:
        print(f"File not found: {e}. Skipping evaluation for this agent.")
        # Optionally, handle or log the exception here

    except Exception as e:
        print(f"An error occurred: {e}. Skipping evaluation for this agent.")
        # Optionally, handle or log the exception here

    return expected_return, expected_action, var, date_list[0]




e_return, e_action, var, start_date =evaluate('NVIV','Data/Industry_56/NVIV/2023-03-09 00_00_00.csv')
print('NVIV: ',e_action,e_return,var, start_date)





