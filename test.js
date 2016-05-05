'use strict';
let Agent = require('./agents/dqnagent.js');
// create an environment object
let env = {};
env.getNumStates = function () {
  return 3;
};
env.getMaxNumActions = function () {
  return 3;
};
// create the DQN agent
let spec = {
  update: 'qlearn', // qlearn | sarsa
  gamma: 0.9, // discount factor, [0, 1)
  epsilon: 0.2, // initial epsilon for epsilon-greedy policy, [0, 1)
  alpha: 0.01, // value function learning rate
  experienceAddEvery: 10, // number of time steps before we add another experience to replay memory
  experienceSize: 5000, // size of experience replay memory
  learningStepsPerIteration: 20,
  tderrorClamp: 1.0, // for robustness
  numHiddenUnits: 100, //
};
let agent = new Agent(env, spec);
let lastPrice = 390.1;
let lastAction = {
  buy: null,
  sell: null
};


let actions = ['wait', 'buy', 'sell'];

const THRESHHOLD = 2;
const INVESTMENT = 1;


let getBuyReward = (price) => {
  let reward = 0;
  if(lastAction.buy){
    lastAction.buy.price = (lastAction.buy.price * lastAction.buy.amount + price * INVESTMENT) / (lastAction.buy.amount + INVESTMENT);
    lastAction.buy.amount += INVESTMENT;
    if(lastAction.buy.amount > THRESHHOLD){
      reward = lastAction.buy.amount * -2;
    }
  }else{
    reward = INVESTMENT * lastAction.sell.price - INVESTMENT * price;
    if (lastAction.sell.amount - INVESTMENT === 0) {
      lastAction.sell = null;
    }
  }
  return reward;
};


let getSellReward = (price) => {
  let reward = 0;
  if(lastAction.sell){
    lastAction.sell.price = (lastAction.sell.price * lastAction.sell.amount + price * INVESTMENT) / (lastAction.sell.amount + INVESTMENT);
    lastAction.sell.amount += INVESTMENT;
    if(lastAction.sell.amount > THRESHHOLD){
      reward = lastAction.sell.amount * -0.5;
    }
  }else{
    reward = INVESTMENT * price - INVESTMENT * lastAction.buy.price;
    if (lastAction.buy.amount - INVESTMENT === 0) {
      lastAction.buy = null;
    }
  }
  return reward;
};


let getReward = (lastAction, newAction, price) => {
  // no action so far taken
  // perform action
  if (!lastAction.buy && !lastAction.sell) {
    if (newAction === 1) {
      lastAction.buy = {
        price: price,
        amount: INVESTMENT
      };
    } else if(newAction === 2){
      lastAction.sell = {
        price: price,
        amount: INVESTMENT
      };
    }
    return 0.01;
  } else if(newAction === 0){
    return 0;
  } else {
    switch (newAction) {
    case 1:
      return getBuyReward(price);
    case 2:
      return getSellReward(price);
    default:
      console.log('this should never happend: ', newAction, lastAction);
      return 0;
    }
  }
};
setInterval(function () { // start the learning loop
  // let newPrice = Math.random() + lastPrice;
  let newPrice = 0.1 + lastPrice; // price is rising
  let diff = newPrice - lastPrice; // is positiv if price is rising
  let s = [newPrice, lastPrice, diff];
  let action = agent.act(s); // s is an array of length 3
  let reward = 0;
  //if action is buy or sell
  if (action === 1 || action === 2) {
    reward = getReward(lastAction, action, newPrice);
    console.log('reward', reward);
  }
  agent.learn(reward); // the agent improves its Q,policy,model, etc. reward is a float
  // console.log('action:', actions[action], ' reward: ', reward, ' newPrice: ', newPrice, ' lastPrice: ', lastPrice);
  lastPrice = newPrice;
}, 10);
