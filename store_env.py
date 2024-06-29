import gym
from gym import spaces
import numpy as np
from datetime import datetime, timedelta
import holidays
import calendar
import random
import matplotlib.pyplot as plt

class StoreEnv(gym.Env):
    def __init__(self, region_param, store_param, product_param_list):
        super(StoreEnv, self).__init__()
       
        # 지역정보 어떤 상권인지, 어떤업종인지에 따라서 유동인구, 인구당유입율, 최대유입율, 휴일활성도 입력
        self.location_type = region_param['location_type']  
        self.floating_population_per_day = region_param['floating_population_per_day']  
        self.attention_ratio = region_param[store_param['store_type']]['attention_ratio']
        self.max_attention = region_param[store_param['store_type']]['max_attention']
        self.holiday_boost = region_param[store_param['store_type']]['holiday_boost']
        self.price_sensitive_customer_ratio = region_param[store_param['store_type']]['price_sensitive_customer_ratio']
       
        # store_param
        self.store_name = store_param['store_name']  
        self.store_address = store_param['store_address']  
        self.store_type = store_param['store_type']
        self.fixed_cost_monthly = store_param['fixed_cost_monthly']
        self.open_hour = store_param['open_hour']  # 0 to 23
        self.close_hour = store_param['close_hour']  # 1 to 24
        self.peak_hour = store_param['peak_hour'] # ex) 12 13 18 19
        self.capital = store_param['capital'] # ex) 12 13 18 19
       
        #sns상의 인기를 시뮬레이션
        self.sns_initial_impression = store_param['sns_impression']
        self.sns_impression = self.sns_initial_impression
        self.sns_impression_min = store_param['sns_impression_min']
        self.sns_impression_max = store_param['sns_impression_max']
        self.sns_simul_drift = store_param['sns_simul_drift']
        self.sns_simul_vol = store_param['sns_simul_vol']
                               
        # product_param_list
        self.products = []
        for product_param in product_param_list:
            product = {
                'product_name': product_param['product_name'],
                'initial_price': product_param['initial_price'],
                'max_price': product_param['max_price'],
                'min_price': product_param['min_price'],
                'cost_per_product': product_param['cost_per_product'],
                'average_market_price': product_param['average_market_price'],
                'initial_inventory_count': product_param['initial_inventory_count']
           }
            self.products.append(product)
       
        #simulation param #그냥 초기값
        self.max_step = 10000
        self.current_step = 0
        self.simul_date = datetime(1999, 1, 1)
        self.simul_date = self.simul_date.replace(hour=self.open_hour, minute=0, second=0)
        self.isholiday = True
 
        self.discount_factor = 0.99  # 0.99로 설정
        self.kr_holidays = holidays.KR()
       
        #파라미터로부터 생성한/할 파라미터
        self.mean_demand =  self.floating_population_per_day * self.attention_ratio
       
        #discrete
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)  # Updated observation shape

        self.initializeSimulation()    
   
    def initializeSimulation(self):
        #총합산 결과 초기화
        self.total_demand = 0
        self.total_sales_count = 0
        self.total_revenue = 0
        self.total_profit = 0
        # 시뮬레이션 결과를 기록할 데이터 구조 초기화
        self.simulation_results = []
        #시뮬레이션 초기화
        self.current_step = 0
        self.sns_impression = self.sns_initial_impression
        #상품별 state 초기화
        self.product_status_list = []
        for product in self.products:
            product_status = {
                                    'current_price': product['initial_price'],
                                    'price_sensitivity' :  product['average_market_price'] / product['initial_price'],
                                    'sales_count': 0,
                                    'inventory_count' : product['initial_inventory_count'],
                                    'revenue': 0,
                                    'profit': 0
                              }
            self.product_status_list.append(product_status)
                           
    def _is_holiday_or_weekend(self, dt):
        return dt.weekday() >= 5 or dt in self.kr_holidays

    def inventory_reset(self):
        for product,product_status in zip(self.products,self.product_status_list):
            product_status['inventory_count'] = product['initial_inventory_count']
   
    def reset(self, current_date, simul_days):
        self.current_step = 0
        self.current_total_demand = 0
        self.total_sales_count = 0
        self.total_revenue = 0
        self.total_profit = 0
        self.simulation_results = []
        #시작일자 지정
        self.simul_date = current_date
        #오픈시간 지정
        self.simul_date = self.simul_date.replace(hour=self.open_hour, minute=0, second=0)
        #날짜 업데이트시
        self.update_holiday()
        self.inventory_reset()
        #혹시 새로운 달로 바뀌었다면 고정비 계산
        self.update_fixedcose_hourly()
        #시뮬레이션 리셋
        self.initializeSimulation()
        #맥스 스텝
        self.max_step = simul_days * (self.close_hour - self.open_hour)   #쉬는시간 있다면 반영 필요.
        return self._get_observation()
   
    def update_fixedcose_hourly(self):
        #시간당 고정비 계산(월별로 일수가 다르니..)
        days_in_month = calendar.monthrange(self.simul_date.year, self.simul_date.month)[1]
        self.fixed_cost_hourly = self.fixed_cost_monthly / days_in_month / (self.close_hour - self.open_hour)    

    def update_holiday(self):
        #휴일체크
        self.isholiday = self._is_holiday_or_weekend(self.simul_date)

    def time_update(self):
        #시간 수정
        if self.simul_date.hour == self.close_hour:
            #새로운 하루      
            self.simul_date += timedelta(days=1)
            self.simul_date = self.simul_date.replace(hour=self.open_hour, minute=0, second=0)
            #날짜 업데이트시
            self.update_holiday()
            self.inventory_reset()
            #혹시 새로운 달로 바뀌었다면 고정비 계산
            self.update_fixedcose_hourly()
        else:
            self.simul_date += timedelta(hours=1)

        return self.simul_date
   
    def policy(self,strategy, q_values=None):
        action = 0
        if strategy == 'close_time_10percent':
            #마감 한시간전 할인
            if self.simul_date.hour == self.close_hour - 1:
                #10% 할인
                action = -2
            else:
                #마감시간이 아니면 초기가격으로
                action = 0
        elif strategy == 'q_learning':
            # Q-Learning을 사용하여 액션 선택하도록 바꿔야 함
            surcharge = random.randint(-4, +4)
            action = surcharge
        else:  #random
            # 랜덤값으로 하나를 한칸은 5%.
            surcharge = random.randint(-4, +4)
            action = surcharge
        return action
    
    def standardPrice_byRound(self, current_price, initial_price, round):
        price = current_price / initial_price * 100
        round * round(price / round)
        return price
    
    def step(self, action):
       
        #시간 업데이트 +1시간 or 다음날
        self.time_update()
       
        for product,product_status in zip(self.products,self.product_status_list):
            # action에 따른 가격 조정. act는 5단위.
            target_percent = 100 + 5 * action
            new_price = product['initial_price'] * target_percent * 0.01
            new_price = np.clip(new_price, product['min_price'], product['max_price']) # min/ max
            product_status['current_price'] = new_price
       
        #푸아송 분포로 방문고객수 시뮬레이션(sns노출, 가격민감도, 휴일, 피크타임 등 가중치 반영 포함)
        self.current_total_demand = self._simulate_demand()

        # 매출 및 수익 계산
        #시간당 전상품 매출 초기화
        sum_sales_count_hourly =0
        sum_revenue_hourly =0
        sum_profit_hourly =0
        for product, product_status in zip(self.products, self.product_status_list):
            sales_count_hourly =0
            revenue_hourly =0
            profit_hourly =0
            #이 부분 개선 필요.... 상품별로 매출을 가능 n등분 해버림pip
            product_demand = self.current_total_demand / len(self.products)
            if(product_status['inventory_count']>product_demand):
                sales_count_hourly = product_demand
            elif(product_status['inventory_count']>0):
                sales_count_hourly = product_status['inventory_count']
            else:
                sales_count_hourly = 0
            revenue_hourly = sales_count_hourly * product_status['current_price']
            profit_hourly = sales_count_hourly * (product_status['current_price'] - product['cost_per_product'])  - self.fixed_cost_hourly

            product_status['sales_count'] += sales_count_hourly
            product_status['inventory_count'] -= sales_count_hourly

            #마감시간에 재고를 비용처리
            if self.simul_date.hour == (self.close_hour -1):
                profit_hourly -= product_status['inventory_count'] * product['cost_per_product']

            product_status['revenue'] += revenue_hourly
            product_status['profit'] += profit_hourly
           
            # 전체 sales, revenue, profit에 합산
            sum_sales_count_hourly += sales_count_hourly
            sum_revenue_hourly += revenue_hourly
            sum_profit_hourly += profit_hourly
   
        # 누적 sales, revenue, profit에 합산
        self.total_sales_count += sum_sales_count_hourly
        self.total_revenue += sum_revenue_hourly
        self.total_profit += sum_profit_hourly
       
        reward = self.total_profit
        discounted_reward = reward * (self.discount_factor ** self.current_step)

        self.current_step += 1
        done = self.current_step >= self.max_step
       
        #파산 체크
        if self.total_profit < -self.capital:
            done = True
       
         # 시뮬레이션 결과를 기록
        simulation_result = {
            'date': self.simul_date,
            'action': action,
            'demand' : self.current_total_demand,
            'sns_impression' : self.sns_impression,
            'profit_hourly' : sum_profit_hourly,
            'total_sales_count': self.total_sales_count,
            'total_revenue': self.total_revenue,
            'total_profit': self.total_profit
        }
        for product, product_status in zip(self.products, self.product_status_list):
            simulation_result[product['product_name'] + '_price'] = product_status['current_price']        
            simulation_result[product['product_name'] + '_inventory_count'] = product_status['inventory_count']        
       
        self.simulation_results.append(simulation_result)
        return self._get_observation(), reward, done, {}

    def _simulate_demand(self):

        #가격 민감도 업데이트
        for product, product_status in zip(self.products, self.product_status_list):
            #1.현재의 민감도 산출 '''
            product_status['price_sensitivity'] = product['average_market_price'] / product_status['current_price']
            '''2.가격에 민감한 고객의 비중을 반영:
             (product_status['price_sensitivity']-1.0)가 가격때문에 줄어들거나 늘어난 고객의 비율이므로
             그 비율에 가격에 민감한 고객에 대한 가정(price_sensitive_customer_ratio)을 곱해서 price_sensitiviy를 최종적으로 산출'''
            product_status['price_sensitivity'] = (product_status['price_sensitivity']-1.0)*self.price_sensitive_customer_ratio + 1.0  
            #영향도를 강화 : 일단 제곱
            product_status['price_sensitivity'] = product_status['price_sensitivity'] * product_status['price_sensitivity']
               
        #sns상의 노출도를 stochastic process로 시뮬레이션 이 결과를 방문고객수에 가중치로 적용
        sns_impact_change = np.random.normal(self.sns_simul_drift, self.sns_simul_vol)
        self.sns_impression = np.clip(self.sns_impression + sns_impact_change, self.sns_impression_min, self.sns_impression_max)
                           
        # Simulate demand as a decreasing function of price
        demand = 0
        # Calculate the price sensitivity factor
        for product, product_status in zip(self.products, self.product_status_list):
            demand += self.mean_demand * np.clip(product_status['price_sensitivity'], 0.01,10) / len(self.products)      

        # Calculate the sns impression factor
        demand = demand * np.clip(self.sns_impression, self.sns_impression_min, self.sns_impression_max)
                   
        if self.isholiday :
            demand = demand * self.holiday_boost
       
        if self.simul_date.hour in self.peak_hour:
            demand = demand * self.max_attention
           
        demand = np.random.poisson(demand)
        return demand

    def _get_observation(self):
        observation = []
        '''
        observation 구조설계
            시간 : 진행율로 표시하자  진행시간/총오픈시간,
            피크타임 여부 : 1, 0
            액션의 평균,
            가격의 평균,
            sns_impression
            수요,
            재고량
            시간당수익
        '''
        if len(self.simulation_results) > 0:
            last_simulation_result = self.simulation_results[-1]
           
            #전처리
            time_date = last_simulation_result['date']
            time_percent = max(time_date.hour-self.open_hour,0) / (self.close_hour - self.open_hour) *100
           
            if self.simul_date.hour in self.peak_hour:
                peak_time = True
            else:
                peak_time = False
           
            price100 =0
            inventory100 =0
            for product, product_status in zip(self.products, self.product_status_list):
                price100     +=  product_status['current_price'] / product['average_market_price'] / len(self.products) *100          
                inventory100 +=   product_status['inventory_count'] / product['initial_inventory_count'] / len(self.products) *100
            observation = [
                            time_percent, #'time_percent'
                            #peak_time, ##'peak_time'         : 
                            #last_simulation_result['sns_impression'], #'sns_impression'    : 
                            price100, # 'price100'
                            last_simulation_result['demand'], #'demand'            : 
                            inventory100, #'inventory100'      : 
                            last_simulation_result['demand'] #'profit_hourly'     : 
                         ]
        else:
            observation = [                            0, #'time_percent'      :
                            #False, ##'peak_time'         :
                            #self.sns_initial_impression ##'sns_impression'    : ,
                            100, #'price100'          : 
                            0,   #'demand'            : 
                            100,  #'inventory100'
                            0   #'profit_hourly'     : 
                          ]
        return observation
           
    def visualize_simulation_results(self):
        simulation_results = self.simulation_results
        dates = [result['date'] for result in simulation_results]
        price = [result['oil_pasta_price'] for result in simulation_results]
        demands = [result['demand'] for result in simulation_results]
        cumul_profit = [result['total_profit'] for result in simulation_results]
        hourly_profit = [result['profit_hourly'] for result in simulation_results]
        sns_impression = [result['sns_impression'] for result in simulation_results]

        plt.figure(figsize=(12, 8))  # 더 넓은 공간으로 확장
        plt.subplot(4, 1, 1)  # 4개의 그래프 중 첫 번째 위치
        plt.plot(dates, cumul_profit, label='Cumulative profit', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Cumulative profit')
        plt.title('Cumulative profit')
        plt.legend()

        plt.subplot(4, 1, 2)  # 4개의 그래프 중 두 번째 위치
        plt.plot(dates, hourly_profit, label='hourly profit', color='blue')
        plt.xlabel('Date')
        plt.ylabel('houly profit')
        plt.title('Houly profit')
        plt.legend()
       
        # Create a secondary y-axis for SNS Impression
        plt2 = plt.gca().twinx()
        plt2.plot(dates, cumul_profit, color='orange')

        # Combine legends from both axes
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = plt2.get_legend_handles_labels()
        plt.gca().legend(lines + lines2, labels + labels2, loc='upper right')
       
        # Set the label for the second y-axis (SNS Impression)
        plt2.set_ylabel('cumul_profit', color='orange', labelpad=10)

        # Subplot 3: Demand and SNS Impression
        plt.subplot(4, 1, 3)
        plt.plot(dates, demands, label='Demand', color='green')
        plt.xlabel('Date')
        plt.ylabel('Demand', color='green', labelpad=10)
        plt.tick_params(axis='y', labelcolor='green')
        plt.title('Demand and SNS Impression Over Time')

        # Create a secondary y-axis for SNS Impression
        plt2 = plt.gca().twinx()
        plt2.plot(dates, sns_impression, color='orange')

        # Combine legends from both axes
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = plt2.get_legend_handles_labels()
        plt.gca().legend(lines + lines2, labels + labels2, loc='upper right')

        # Set the label only for the first y-axis (Demand)
        plt.ylabel('Demand', color='green')

        # Set the label for the second y-axis (SNS Impression)
        plt2.set_ylabel('SNS Impression', color='orange', labelpad=10)

        plt.subplot(4, 1, 4)  # 4개의 그래프 중 세 번째 위치
        plt.plot(dates, price, label='Price', color='red')  # Price 그래프 추가
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Price Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()

