t = require 'torch'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'



--require 'strict'

package.cpath = package.cpath .. ";/home/zdevito/terra/release/lib/?.so"

require 'terra'

runN = 0

assert(terralib.loadfile("funMap.t"))()
local doneFuns = {}

--doneFuns["util.typeAsInPlace"] = true
--doneFuns["torch.DoubleTensor.expandAs"] = true
doneFuns["torch.pow"] = true
--doneFuns["torch.t"] = true
doneFuns["torch.max"] = true
doneFuns["torch.log"] = true
doneFuns["torch.exp"] = true
--util.fillInPlace
doneFuns["torch.neg"] = true
doneFuns["torch.sum"] = true
doneFuns["torch.eq"] = true
doneFuns["torch.cdiv"] = true
doneFuns["torch.add"] = true
doneFuns["torch.mm"] = true
doneFuns["torch.cmul"] = true

local funMap = {}

function torchWrap(name,fun)
  --if(name=="torch.t") then
  --  fun = "torch.t"
  --end
  --if(name=="util.typeAsInPlace") then
  --  fun = "util.typeAsInPlace"
  --end
  
  return function(...) 
    if doneFuns[name] then
      --print("DoneFun",name)
      local args = {...}
      local argTypes = {}
      local argTypeStr = ""
      for i,arg in ipairs(args) do
        local typeStr = ""
        if(torch.isTensor(arg)) then
          typeStr = arg:type()
        elseif(type(arg)=="number") then
          typeStr = "number"
        else 
          assert(false,"Bad type! arg"..i..": "..torch.type(arg))
        end
        table.insert(argTypes,typeStr)
        argTypeStr = argTypeStr .. typeStr
      end
      if funMap[fun] == nil then
        funMap[fun] = {}
      end
      if funMap[fun][argTypeStr] == nil then
        print("CREATING FUNCTION for ",name,argTypeStr)
        funMap[fun][argTypeStr] = createFun(fun,argTypes)
        print("YAY, MADE FUNCTION")
      end
      return funMap[fun][argTypeStr](...)
    else
      return fun(...)
    end
  end

end


grad = require 'autograd'

geometry = {32,32}
xSize = 32*32
numClasses = 10
batchSize = 32
trainSize = 64000
testSize = 16*100
lr = .01

trainMnist = (runN <2)
epochs=4



trainData = mnist.loadTrainSet(trainSize, numClasses)
trainData:normalizeGlobal(mean, std)
testData = mnist.loadTestSet(testSize, numClasses)
testData:normalizeGlobal(mean, std)



--x: (batchsize x 1024)
--y: (batchsize x 10) (1-hot)
net = function(params, x,y)
  local h1 = x*params.W[1]
  h1 = h1 + params.b[1]:expandAs(h1)
  
  local max = t.max(h1,2)
  local exp = t.exp(h1-max:expandAs(h1))
  local sum = t.sum(exp,2)
  local yProb = t.cdiv(exp,sum:expandAs(exp))
  
  local loss = t.sum(-t.log(t.sum(t.cmul(yProb,y),2)))/batchSize
  --print(h1,max,exp,sum,yProb,loss)
  return loss
end


netForward = function(params, x)
  local h1 = x*params.W[1]
  h1 = h1 + params.b[1]:expandAs(h1)
  
  local max = t.max(h1,2)
  local exp = t.exp(h1-max:expandAs(h1))
  local sum = t.sum(exp,2)
  local yProb = t.cdiv(exp,sum:expandAs(exp))
  
  return yProb
end

dnet = grad(net,{optimize=true})

--initialize parameters
params = {
  W = {
    t.randn(xSize,numClasses)-0.5
  },
  b = {
    t.zeros(1,numClasses)
  }
}

if(runN >0) then
  x = t.reshape(trainData.data[{{1,1+batchSize-1}}],batchSize,xSize)
  yLabels = trainData.labels[{{1,1+batchSize-1}}]:long():view(batchSize,1)
  y = t.zeros(batchSize,numClasses)
  y:scatter(2,yLabels,1)

  for i=1,runN do
    dparams, loss = dnet(params,x,y)
  end
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
      if((tr-1)%49==0) then
        print(e,tr,loss)
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
  print(numCorrect/testData.size())
end

