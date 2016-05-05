'use strict';
// Utility fun
let assert = (condition, message) => {
  // from http://stackoverflow.com/questions/15313418/javascript-assert
  if (!condition) {
    message = message || 'Assertion failed';
    if (typeof Error !== 'undefined') {
      throw new Error(message);
    }
    throw message; // Fallback
  }
};
// Random numbers utils
let returnV = false;
let vVal = 0.0;
let gaussRandom = () => {
  if (returnV) {
    returnV = false;
    return vVal;
  }
  let u = 2 * Math.random() - 1;
  let v = 2 * Math.random() - 1;
  let r = u * u + v * v;
  if (r === 0 || r > 1) {
    return gaussRandom();
  }
  let c = Math.sqrt(-2 * Math.log(r) / r);
  vVal = v * c; // cache this
  returnV = true;
  return u * c;
};
let randf = (a, b) => {
  return Math.random() * (b - a) + a;
};
let randi = (a, b) => {
  return Math.floor(Math.random() * (b - a) + a);
};
let randn = (mu, std) => {
  return mu + gaussRandom() * std;
};
// helper function returns array of zeros of length n
// and uses typed arrays if available
let zeros = (n) => {
  if (typeof (n) === 'undefined' || isNaN(n)) {
    return [];
  }
  if (typeof ArrayBuffer === 'undefined') {
    // lacking browser support
    let arr = new Array(n);
    for (let i = 0; i < n; i++) {
      arr[i] = 0;
    }
    return arr;
  } else {
    return new Float64Array(n);
  }
};

// Mat utils
// fill matrix with random gaussian numbers
let fillRandn = (m, mu, std) => {
  for (var i = 0, n = m.w.length; i < n; i++) {
    m.w[i] = randn(mu, std);
  }
};

let fillRand = (m, lo, hi) => {
  for (var i = 0, n = m.w.length; i < n; i++) {
    m.w[i] = randf(lo, hi);
  }
};

let gradFillConst = (m, c) => {
  for (var i = 0, n = m.dw.length; i < n; i++) {
    m.dw[i] = c;
  }
};



  let sig = (x) => {
    // helper function for computing sigmoid
    return 1.0/(1+Math.exp(-x));
  };


  /*
  Mat holds a matrix
  **/
let Mat = function (n, d) {
  // n is number of rows d is number of columns
  this.n = n;
  this.d = d;
  this.w = zeros(n * d);
  this.dw = zeros(n * d);
};

Mat.prototype = {
  get: function (row, col) {
    // slow but careful accessor function
    // we want row-major order
    let ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    return this.w[ix];
  },
  set: function (row, col, v) {
    // slow but careful accessor function
    let ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    this.w[ix] = v;
  },
  setFrom: function (arr) {
    for (let i = 0, n = arr.length; i < n; i++) {
      this.w[i] = arr[i];
    }
  },
  setColumn: function (m, i) {
    for (let q = 0, n = m.w.length; q < n; q++) {
      this.w[(this.d * q) + i] = m.w[q];
    }
  },
  toJSON: function () {
    return {
      n: this.n,
      d: this.d,
      w: this.w,
    };
  },
  fromJSON: function (json) {
    this.n = json.n;
    this.d = json.d;
    this.w = zeros(this.n * this.d);
    this.dw = zeros(this.n * this.d);
    for (let i = 0, n = this.n * this.d; i < n; i++) {
      this.w[i] = json.w[i]; // copy over weights
    }
  }
};
// return Mat but filled with random numbers from gaussian
let RandMat = function(n, d, mu, std) {
  var m = new Mat(n, d);
  fillRandn(m, mu, std);
  //fillRand(m,-std,std); // kind of :P
  return m;
};

let copyMat = (b) => {
  let a = new Mat(b.n, b.d);
  a.setFrom(b.w);
  return a;
};

let copyNet = function (net) {
  // nets are (k,v) pairs with k = string key, v = Mat()
  let newNet = {};
  for (let p in net) {
    if (net.hasOwnProperty(p)) {
      newNet[p] = copyMat(net[p]);
    }
  }
  return newNet;
};
let updateMat = function (m, alpha) {
  // updates in place
  for (let i = 0, n = m.n * m.d; i < n; i++) {
    if (m.dw[i] !== 0) {
      m.w[i] += -alpha * m.dw[i];
      m.dw[i] = 0;
    }
  }
};
let updateNet = function (net, alpha) {
  for (let p in net) {
    if (net.hasOwnProperty(p)) {
      updateMat(net[p], alpha);
    }
  }
};
let netToJSON = (net) => {
  let j = {};
  for (let p in net) {
    if (net.hasOwnProperty(p)) {
      j[p] = net[p].toJSON();
    }
  }
  return j;
};
let netFromJSON = (j) => {
  let net = {};
  for (let p in j) {
    if (j.hasOwnProperty(p)) {
      net[p] = new Mat(1, 1); // not proud of this
      net[p].fromJSON(j[p]);
    }
  }
  return net;
};
let netZeroGrads = (net) => {
  for (let p in net) {
    if (net.hasOwnProperty(p)) {
      let mat = net[p];
      gradFillConst(mat, 0);
    }
  }
};
let netFlattenGrads = (net) => {
  let n = 0;
  for (let p in net) {
    if (net.hasOwnProperty(p)) {
      let mat = net[p];
      n += mat.dw.length;
    }
  }
  let g = new Mat(n, 1);
  let ix = 0;
  for (let p in net) {
    if (net.hasOwnProperty(p)) {
      let mat = net[p];
      for (let i = 0, m = mat.dw.length; i < m; i++) {
        g.w[ix] = mat.dw[i];
        ix++;
      }
    }
  }
  return g;
};

/*
  Agent util functions
**/

// syntactic sugar function for getting default parameter values
let getopt = function(opt, fieldName, defaultValue) {
  if(typeof opt === 'undefined') { return defaultValue; }
  return (typeof opt[fieldName] !== 'undefined') ? opt[fieldName] : defaultValue;
};

let setConst = function(arr, c) {
  for(let i=0,n=arr.length;i<n;i++) {
    arr[i] = c;
  }
};

let sampleWeighted = function(p) {
  let r = Math.random();
  let c = 0.0;
  for(let i=0,n=p.length;i<n;i++) {
    c += p[i];
    if(c >= r) { return i; }
  }
  assert(false, 'wtf');
};
/*
  a module for all the util functions
**/
module.exports = {
  Mat: Mat,
  RandMat: RandMat,
  assert: assert,
  sig: sig,
  updateMat: updateMat,
  updateNet: updateNet,
  copyNet: copyNet,
  copyMat: copyMat,
  netToJSON: netToJSON,
  netFromJSON: netFromJSON,
  netZeroGrads: netZeroGrads,
  netFlattenGrads: netFlattenGrads,
  getopt: getopt,
  setConst: setConst,
  sampleWeighted: sampleWeighted,
  randi: randi
};
