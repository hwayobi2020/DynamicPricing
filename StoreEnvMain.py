import numpy as np
from datetime import datetime
from store_env import StoreEnv
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

region_param = {
                'location_type': "working area",
                'floating_population_per_day': 1000,
                'restaurant': {
                                'attention_ratio': 0.0046,
                                'max_attention': 5,
                                'holiday_boost': 0.9,
                                'price_sensitive_customer_ratio' : 0.99
                            },
                'gym':        {
                                'attention_ratio': 0.0001,
                                'max_attention': 3,
                                'holiday_boost': 0.3,
                                'price_sensitive_customer_ratio' : 0.99
                            },
                'hospital':   {
                                'attention_ratio': 0.003,
                                'max_attention': 5,
                                'holiday_boost': 0.3,
                                'price_sensitive_customer_ratio' : 0.99
                            },
                'coffee':    {
                                'attention_ratio': 0.005,
                                'max_attention': 5,
                                'holiday_boost': 0.3,
                                'price_sensitive_customer_ratio' : 0.99
                            }
                }

store_param = {
                    'store_name' : 'diamond pasta',
                    'store_address': "makokdong-ro 12, gangseo-gu, seoul",
                    'store_type': "restaurant",      #restuarant, gym, hospital, coffee
                    'fixed_cost_monthly': 1000,
                    'open_hour': 10,
                    'close_hour': 22,
                    'peak_hour' : {12,13,18,19},
                    'capital'   : 10000,   #자본여력, 이 이상이라면 파산. gym done
                    'sns_impression' : 1,  # 1~10 사이의 값.
                    'sns_simul_drift' : 0.0001,
                    'sns_simul_vol' : 0.002,
                    'sns_impression_min' : 0.8,
                    'sns_impression_max' : 1.2
                }

product_param_list = [
                        {
                            'product_name': 'oil_pasta',
                            'initial_price': 10,
                            'max_price': 15,
                            'min_price': 6,
                            'cost_per_product': 6.1,
                            'average_market_price' : 10,
                            'initial_inventory_count': 200
                        }
                    ]

import numpy as np
import tensorflow as tf
from datetime import datetime


def DQN_learning():
    num_states = 5  # Number of state features (time_percent, price100, demand, inventory100)
    num_actions = 9  # Number of possible actions (-4 to 4)

    # Q-Network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(num_states + 1,)),  # Modified input shape
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_actions)  # Q-value output
    ])
    
    # Define an optimizer (e.g., Adam optimizer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Q-learning parameters
    discount_factor = 0.99
    exploration_prob = 0.2  # Exploration probability

    # Create an environment object
    env = StoreEnv(region_param, store_param, product_param_list)  # Assuming you have defined StoreEnv

    num_episodes = 10  # Number of training episodes

    for episode in range(num_episodes):
        current_date = datetime(2023, 9, 1)
        total_days = 10
        state = env.reset(current_date, total_days)
        done = False

        while not done:
            if np.random.rand() < exploration_prob:
                # Random action selection (-4 to 4)
                action = np.random.randint(0, num_actions) - 4
            elif state is None:
                # Random action selection (-4 to 4)
                action = np.random.randint(0, num_actions) - 4
            else:
                # Q-Learning based action selection
                action = 0  # Placeholder action
                state_expanded = np.append(state, action).reshape(1, -1)  # Append the action to the state and reshape
                #state_expanded = np.array([8.333333333333332, 105.0, 3.0, 98.5, 3.0, 1.0]).reshape(1, -1)
                q_values = model.predict(state_expanded)
                action = np.argmax(q_values) - 4  # Convert action index to actual action (-4 to 4)
                
            next_state, reward, done, _ = env.step(action)  # Action is now a list with one element

            # Grad code
            with tf.GradientTape() as tape:
                state_expanded = np.append(next_state, action).reshape(1, -1)  # Append the action to the state and reshape
                q_values = model(state_expanded)  # Use the model directly
                next_max_q = np.argmax(q_values) - 4
                target_q = reward + discount_factor * next_max_q
                loss = tf.reduce_mean(tf.square(q_values - target_q))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            state = next_state

     # Save the trained model
    model.save('D:\gpt_code\pricing')
    
    print("Q-Network Model Summary:")
    model.summary()
    
    # Visualize simulation results
    env.visualize_simulation_results()
   
def main():
   
    env = StoreEnv(region_param, store_param, product_param_list)

    current_date = datetime(2023, 9, 1)
    total_days = 30
    state = env.reset(current_date, total_days)  
   
    # 시뮬레이션을 수행
    done = False
    total_reward = 0
    while not done:
        # 랜덤하게 액션 선택 (0에서 10 사이의 값)
        #action = env.policy('close_time_10percent')
        action = env.policy('random')

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        print(f"Date: {current_date}")
        print(f"  Current State: {state}")
        print(f"  Action Taken: {action}")
        # for product, product_status in zip(env.products, env.product_status_list):
        #     print(f"  current price: {product_status['current_price']}")
        #     print(f"  sales_count: {product_status['sales_count']}")
        #     print(f"  revenue: {product_status['revenue']}")
        #     print(f"  inventory_count: {product_status['inventory_count']}")
        #     print(f"  profit: {product_status['profit']}")
        print(f"  Done: {done}\n")

        state = next_state
       
    #시각화
    env.visualize_simulation_results()
    
def qmain():
    # Train the DQN
    #DQN_learning()

    # Load the trained Q-Network model from the SavedModel format
    model = tf.keras.models.load_model('D:/gpt_code/pricing/')

    # Simulate and visualize using the trained Q-Network
    env = StoreEnv(region_param, store_param, product_param_list)
    current_date = datetime(2023, 9, 1)
    total_days = 30
    state = env.reset(current_date, total_days)

    # Simulate using the trained Q-Network
    done = False
    total_reward = 0
    while not done:
        # Use Q-network for action selection
        state_expanded = np.append(state, 0).reshape(1, -1)  # Placeholder action (0) for Q-network
        q_values = model.predict(state_expanded)
        action = np.argmax(q_values) - 4

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        print(f"Date: {current_date}")
        print(f"  Current State: {state}")
        print(f"  Action Taken: {action}")
        # Rest of your print statements...

        state = next_state

    # Visualize simulation results
    env.visualize_simulation_results()
    
    # Print the total reward obtained during the simulation
    print("Total Reward:", total_reward)

if __name__ == "__main__":
    #main()
    DQN_learning()
    #qmain()

