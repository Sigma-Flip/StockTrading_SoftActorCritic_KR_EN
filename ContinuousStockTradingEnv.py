
import numpy as np


class ContinuousStockTradingEnv():
    '''
    ### description 
    
    특정 주식의 뉴스발표일을 기준으로 최적의 long/short 전략을 도출하는 에이전트를 학습하기 위한 환경입니다. 
    논문에서 쓰이는 대부분의 에이전트는 100% buy & 100% sell, 즉 discrete한 action만이 가능하다는 한계와
    에이전트가 하나의 주식차트만을 학습하기 때문에 확장가능성이 작다는 단점이 있었습니다. 
    
    이를 해당 모델에서는 62% buy와 같이 연속적인 action을 구현함으로써, 좀 더 현실적인 모델을 만들었습니다. 또한 모델을 확장시켜 multi stock model
    을 구현함으로써 각자 포트폴리오에 대한 최적 전략도 찾을 수 있게 할 예정입니다. 무엇보다 단순한 상황이 아닌, 레포트나 뉴스등이 나온 후 30일 만큼만
    학습을 시켰기 때문에, 특히 뉴스 이벤트 period에 특화된 모델링이 가능하다는 점이 가장 큰 매력입니다. 
    
    '''

    '''
    제작자 : 이동우
    ver : 0 ( 단일 stock ver)
    '''

    '''
    ### Observation Space
    
    observation : 'ndarray' with shape (2,)
    
    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | money Value                          | 0    | Inf | position (m) |
    | 1   | Stock Value                          | 0    | Inf | position (m) |
    
    
    ### Action Space
    
    Trade Action : 'ndarray' with shape(1,), action은 [-1,1]로 제한되어 있으며 
                    {'-1' = '100% sell' , '+1' = 100% buy} 를 의미합니다. 
                    
    ### Transition Dynamics
    
    Agent가 특정 action(buy & sell)을 수행한 후에는 환경에서 바뀐 observaton(상태)를 반환해줍니다. 
    
    1st. Stocks Value <- Stocks Value + (Max_trading_units) * (price) * (action)
    2nd. Money Value <- Money Value - (Max_trading_units) * (price) * (action)  
    
    
    ### Total Reward
    주식 트레이드에서 보상은 벌어들인 수익입니다. 이는 episode가 끝났을 때의 최종 가치에서 초기 가치를 뺀 값으로 추정합니다. 
    (단, 각 매매는 수수료를 발생시키기 때문에 이를 penalty로 적용합니다.)
    
    Reward = MoneyValue + StockValue - (Initial_MoneyValue + Initial_StockValue)
    (For each step, Reward -= Trading Volume * TransactionCost)
    
    * TradingVolume = (Max_trading_unit) * (Price) * abs(action)
    if Trading Volume = 0 => 그 날은 매매를 관망(Hold) => 수수료 0 부과
    
    ### Terminal State
    
    1st. Reward가 K 이상으로 넘어갈 때( K = HyperParameter)
    2nd. episode의 길이가 한달을 넘어가면 종료 (뉴스 이벤트의 영향이 사라지기 때문에) 
    
    ### Arguments
    
    '''
    # gym.make('Continuous_StockTradingEnv-v0')
    '''

    ### Verision History

    * v0 : Initial Ver (단일 주식 매매프로그램)
    * v1(예정) : Multi Stock Trading Agent
    '''
    def __init__(self,chart_data):

        self.initial_mv = 1000000
        self.initial_sv = 0
        self.mv = self.initial_mv
        self.sv = self.initial_sv
        self.num_stocks = 0
        self.balance = 0
        self.transaction_cost = 0.01
        self.chart_data = chart_data

        self.min_action = -1.0
        self.max_action = 1.0

        self.min_mv = 0  # mv = Money Value
        self.max_mv = 10000000000
        self.min_sv = 0  # sv = Stock Value
        self.max_sv = 10000000000
        self.goal_balance = self.initial_mv * 2
        self.temporal_reward = []
        self.cumulative_reward = []
        self.n_step = 0
        self.max_step = 29

        self.state_ohlcv, self.next_state_ohlcv, self.price, self.next_price = self.get_ohlcv_state()


        self.low_state = np.array(
            [self.min_mv, self.min_sv], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_mv, self.max_sv], dtype=np.float32
        )

        self.action_space_low = self.min_action
        self.action_space_high = self.max_action
        self.observation_space_low = self.low_state
        self.observation_space_high = self.high_state


    def get_ohlcv_state(self):
        state = self.chart_data.iloc[self.n_step].to_dict()
        next_state = self.chart_data.iloc[self.n_step+1].to_dict()
        price = state['Adj Close']
        next_price = next_state['Adj Close']

        return state, next_state, price, next_price

    def decide_trading_units(self,action):
        if action >0:
            max_buyable_units = int(self.mv / (self.price*(1+self.transaction_cost)) )
            max_sellable_units = 0
        elif action <= 0:
            max_buyable_units = 0
            max_sellable_units = self.num_stocks

        return max_buyable_units, max_sellable_units



    def step_(self, action):

        self.state_ohlcv, self.next_state_ohlcv, self.price, self.next_price = self.get_ohlcv_state()
        max_buyable_units, max_sellable_units = self.decide_trading_units(action)
        # print('step: ', self.n_step)
        # print('max buyable units:',max_buyable_units)
        # print('max sellable units:',max_sellable_units)
        # print('mv : ', self.mv)
        # print('sv : ', self.sv)


        sv = self.sv
        mv = self.mv
        self.balance = sv + mv
        force = min(max(action[0], self.min_action), self.max_action)

        buying_units = int(max_buyable_units * force)
        selling_units= int(max_sellable_units * abs(force))
        trading_volume = (buying_units + selling_units) * self.price


        self.num_stocks += buying_units - selling_units
        self.sv =  (self.num_stocks * self.next_price) - trading_volume * self.transaction_cost
        self.mv += -(buying_units * self.price) + (selling_units * self.price)
        self.balance_ = self.sv + self.mv

        terminated = bool(
            (self.balance_ >= self.goal_balance)  or (self.n_step==28)
        )




        temporal_reward = self.balance_ - self.balance
        self.temporal_reward.append(temporal_reward)

        # cumulative_reward = self.temporal_reward.sum()
        # self.cumulative_reward.append(cumulative_reward)


        # reward = 0
        # reward = self.mv + self.sv - (self.initial_sv + self.initial_mv)
        # reward -= trading_volume * self.transaction_cost
        # print(self.next_state_ohlcv)
        ohlcv_  = list(self.next_state_ohlcv.values())
        position  = [self.mv ,self.sv, self.balance_, action[0], self.num_stocks]
        self.state_ = np.array(ohlcv_ + position)
        # print('self.state : ',self.state_)
        self.n_step += 1

        return self.state_, temporal_reward, terminated, False


    def reset(self):
        # super().reset(seed=seed)

        self.mv = self.initial_mv
        self.sv = self.initial_sv
        self.num_stocks = 0
        self.balance = 0
        self.n_step = 0
        self.temporal_reward = []
        self.cumulative_reward = []

        self.state_ohlcv, self.next_state_ohlcv, self.price, self.next_price = self.get_ohlcv_state()
        self.state = np.array([self.mv, self.sv, self.balance, 0, self.num_stocks] + list(self.state_ohlcv.values()))

        # print('state:',self.state)
        return np.array(self.state, dtype=np.float32)