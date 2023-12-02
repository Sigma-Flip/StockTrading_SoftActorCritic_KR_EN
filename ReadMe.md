본 파일은 연속적인 주식거래를 학습시킵니다. 

< 데이터 > 
1. Ohlcv, industry : FinanceDataReader에서 추출
2. 그 외 : 이동평균 관련 지수들 (필요시 삭제 및 타 변수 추가 가능)
   
<main> - 실행 o
1. Train_1Agent.py : 차트 1개에 대해서 학습을 시키는 모델입니다. n_games로 몇 번 반복학습을 시킬지 설정할 수 있으며 plots에서 최종 score를 확인가능합니다. 
2. Evaluate_1Agent : Train_1Agent.py 에서 학습을 사전학습을 시켜놓은 모델을 테스트합니다. 
3. Train_N_Agent.py : n개의 차트에 대해서 학습을 시키는 모델입니다. 저는 utils.py - get_file_path를 활용하여 여러 개의 차트를 가져왔습니다. 그 후는 TrainAgnet.py와 크게 다르지 않습니다. 
4. Evaluate_N_Agent: Train_N_Agent.py에서 사전학습시킨 각각의 차트에 대한 모델들을 test데이터에서 실행시킵니다. n개의 모델들이 1개의 test chart에 대해 action을 하기 때문에 각ㅁ 날짜에 대한 평균 action, 평균 보상, 보상에 대한 분산을 csv파일로 반환해줍니다. csv파일은 각 정보들이 n개의 agent들에 대한  평균값을 반환하기 때문에 N_agents들 각각의 action과 reward를 확인하려면 plots/ticker 로 들어가면 확인할 수 있습니다. 

* 실행 방법 :
* 1)Train_1Agent.py -> Evaluate_1Agent 
* 2) Train_N_Agent.py -> Evaluate_N_Agent

< 환경설정 >  - 실행x ( 하이퍼 파라미터 제외 )  
1. buffer.py : ReplayBuffer로 각 (State, Action, Reward, Next State) 를 기록합니다. 
2. networks.py : 강화학습에 필요한 네트워크들을 정의하고 있습니다. 해당 모델은 Soft Actor Critic 이기 때문에 [Criticnetwork, ActorNetwork, ValueNetwork]에 대한 정보가 들어 있습니다. 
3. sac_torch.py : 각각의 네트워크들이 상호작용하여 실제로 Action을 취하는 agent를 정의하고 있습니다. 각각의 네트워크들이 어떻게 상호작용하고 있는지에 대한 정보와 각종 hyper parameter들에 대한 정보가 들어 있습니다. 
                  SAC의 가장 큰 특징인 엔트로피도 해당 코드에서 정의되어 있습니다.
4. ContinuousStockTradingEnv.py  : agent가 실제 행동을 하기 위한 환경을 정의하고 있습니다. 실제 수익을 보상으로 반환해주며, 거래 수수료를 페널티로 설정하였습니다. 자세한 설명은 주석에서 확인 가능합니다. 
5. utils.py : 시각화, 랜덤추출등의 코드

< Reference> - 이 두 코드에서 큰 도움을 얻었습니다. 
1. Model - https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC
2. Env -https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
gym/gym/envs/classic_control/continuous_mountain_car.py at master · openai/gym



This file learns continuous stock trading. 

<Data > 
1. 1. Ohlcv, industry : FinanceDataReader에서 추출
2. Others: Indexes related to moving averages (deleted and added other variables if necessary)
   
<main> - Run o
1. Train_1Agent.py : This model allows you to learn about one chart. You can set how many repetitions you want to do with n_games and check the final score on the plots. 
2. Evaluate_1Agent : Train_1Agent.py tests the pre-learning model. 
3. Train_N_Agent.py : It is a model that learns about n charts. I brought several charts using utils.py - get_file_path. After that, it is not much different from TrainAgnet.py . 
4. In Evaluate_N_Agent: Train_N_Agent.py, run the models for each chart pre-trained in the test data. Since n models act on one test chart, they return the average action, average reward, and variance for each ㅁ date to the csv file. The csv file returns the average value of each of the n agents, so you can check the action and reward of each of the N_agent by entering the plots/ticker. 

* How to do this:
* * 1)Train_1Agent.py -> Evaluate_1Agent 
* * 2) Train_N_Agent.py -> Evaluate_N_Agent

< Configuration > - Run x (excluding hyperparameters)  
1. buffer.py : Record each (State, Action, Reward, Next State) with ReplayBuffer. 
2. networks.py : It defines the networks needed for reinforcement learning. Because the model is Soft Actor Critical, it contains information about Critical Network, ActorNetwork, ValueNetwork. 
3. sac_torch.py : defines an agent that each network interacts with and actually takes action on. It contains information about how each network is interacting with, and information about various hyperparameters. 
                  Entropy, the biggest feature of the SAC, is also defined in its code.
4. ContinuousStockTradingEnv.py : The agent is defining an environment for actual action. It returns actual profits as compensation, and the transaction fee is set as a penalty. You can find a detailed explanation in the comments. 
5. utils.py : Code such as visualization, random extraction, etc

<Reference> - These two codes helped us greatly. 
1. 1. Model - https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC
2. 2. Env -https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
gym/gym/envs/classic_control/continuous_mountain_car.py at master · openai/gym
