import pandas as pd
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve, get_filepath
from ContinuousStockTradingEnv import ContinuousStockTradingEnv


if __name__ == '__main__':

    df = pd.read_csv('./Data/Industry_1/TPR/2023-06-22 00_00_00.csv')
    df.drop(['industry','event'],axis=1, inplace=True)

    '''
    <주식환경 파트>
    env : 주식매매환경 임포트
    agent : 주식투자를 실제로 실행하고 평가하는 에이전트 생성
    n_games : 수행할 게임(episodes) 개수
    n_count : 1일차 ~ 30일차까지 iteration
    '''
    env = ContinuousStockTradingEnv(df)
    agent = Agent(input_dims=[14], env=env,
            n_actions=1)
    n_games = 10000 # 하이퍼파리미터, 반복할 게임 학습 수

    n_count = 0

    '''<성과측정 파트> - 성과 그래프 저장할 디렉토리 지정 및 각 에피소드마다의 score저장 '''
    filename = 'StockTrade.png'
    figure_file = 'plots/' + filename

    best_score = 0
    score_history = []
    action_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')

    '''<학습파트> - n_games 만큼의 episodes 학습'''

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:

            action = agent.choose_action(observation)
            # print('episode : ', i)
            # print('action: ', action)

            observation_, reward, done, info = env.step_(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)

            if not load_checkpoint:
                agent.learn()

            observation = observation_
            # n_count += 1
            print()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x,score_history, figure_file)



