this is a try to modularize the original code from:
## [REINFORCE.js](http://cs.stanford.edu/people/karpathy/reinforcejs/ "REINFORCE.js Demo Page")
### see also: [Original Github Repo](https://github.com/karpathy/reinforcejs "Github Repo")



  ```javascript
  let reinforce = require('reinforcenode');
  let Agent = reinforce.DQNAgent;
  // create an environment object
  let env = {};
  env.getNumStates = function() { return 8; }
  env.getMaxNumActions = function() { return 4; }

  // create the DQN agent
  let spec = { alpha: 0.01 } // see full options on DQN page
  agent = new Agent(env, spec);

  setInterval(function(){ // start the learning loop
    let action = agent.act(s); // s is an array of length 8
    //... execute action in environment and get the reward
    agent.learn(reward); // the agent improves its Q,policy,model, etc. reward is a float
  }, 0);
  ```
