t = require 'torch'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

package.cpath = package.cpath .. ";/home/zdevito/terra/release/lib/?.so"
assert(require 'terra')
package.terrapath = package.terrapath .. ";/home/rdaly525/autodiff/src/?.t"

require 'trace'

useTerra = false
runN = 0

grad = require 'autograd'

xSize = 32*32
hiddenSizes = {12}
numClasses = 10
batchSize = 32
trainSize = 64000
testSize = 16*100
lr = .01

trainMnist = (runN <2)
epochs=10

trainData = mnist.loadTrainSet(trainSize, numClasses)
trainData:normalizeGlobal(mean, std)
testData = mnist.loadTestSet(testSize, numClasses)
testData:normalizeGlobal(mean, std)


--one layer classification using softmax loss
--x: (batchsize x 1024)
--y: (batchsize x 10) (1-hot)


--initialize parameters
params = {
  W = {
    t.randn(xSize,hiddenSizes[1])-0.5,
    t.randn(hiddenSizes[1],numClasses)-0.5
  },
  b = {
    t.zeros(1,hiddenSizes[1]),
    t.zeros(1,numClasses)
  }
}

net = function(params, x,y)
  local h1 = x*params.W[1]
  h1 = t.tanh(h1 + params.b[1]:expandAs(h1))
  local h2 = h1*params.W[2]
  h2 = h2 + params.b[2]:expandAs(h2)

  --log softmax
  local max = t.max(h2,2)
  local exp = t.exp(h2-max:expandAs(h2))
  local sum = t.sum(exp,2)
  local yProb = t.cdiv(exp,sum:expandAs(exp))
  
  --Regularization
  local reg = 0
  for i=1,2 do
    reg = reg+t.sum(t.cmul(params.W[i],params.W[i]))
  end
 
  local loss = t.sum(-t.log(t.sum(t.cmul(yProb,y),2)))/batchSize + .01*reg
  --print(h1,max,exp,sum,yProb,loss)
  return loss
end


netForward = function(params, x)
  local h1 = x*params.W[1]
  h1 = t.tanh(h1 + params.b[1]:expandAs(h1))
  local h2 = h1*params.W[2]
  h2 = h2 + params.b[2]:expandAs(h2)
  
  local max = t.max(h2,2)
  local exp = t.exp(h2-max:expandAs(h2))
  local sum = t.sum(exp,2)
  local yProb = t.cdiv(exp,sum:expandAs(exp))
  
  return yProb
end

dnet = grad(net,{optimize=true})



if(runN >0) then
  x = t.reshape(trainData.data[{{1,1+batchSize-1}}],batchSize,xSize)
  yLabels = trainData.labels[{{1,1+batchSize-1}}]:long():view(batchSize,1)
  y = t.zeros(batchSize,numClasses)
  y:scatter(2,yLabels,1)

  t0 = os.time()
  for i=1,runN do
    dparams, loss = dnet(params,x,y)
  end
  tN = os.time()
  print("Total for "..runN.." is "..(tN-t0) .. " seconds.")
  print("seconds per grad is "..((tN-t0)/runN))
end

if(trainMnist) then
  for e=1,epochs do
    for tr=1,trainData:size(),batchSize do
      x = t.reshape(trainData.data[{{tr,tr+batchSize-1}}],batchSize,xSize)
      yLabels = trainData.labels[{{tr,tr+batchSize-1}}]:long():view(batchSize,1)
      y = t.zeros(batchSize,numClasses)
      y:scatter(2,yLabels,1)

      dparams, loss = dnet(params,x,y)
      params.W[1]:add(-lr,dparams.W[1])
      params.b[1]:add(-lr,dparams.b[1])
      params.W[2]:add(-lr,dparams.W[2])
      params.b[2]:add(-lr,dparams.b[2])
      if((tr-1)%(batchSize*50)==0) then
        print(e,tr-1,loss)
      end
    end
  end

  local numCorrect = 0
  for tr=1,testData:size(),batchSize do
    x = t.reshape(testData.data[{{tr,tr+batchSize-1}}],batchSize,xSize)
    yLabels = testData.labels[{{tr,tr+batchSize-1}}]:long():view(batchSize,1)
    y = t.zeros(batchSize,numClasses)
    y:scatter(2,yLabels,1)

    yProb = netForward(params,x,y)
    maxs, idxs = t.max(yProb,2)
    numCorrect = numCorrect + t.sum(t.eq(idxs,yLabels))
  end
  print("Accuracy:", numCorrect/testData.size())
end

