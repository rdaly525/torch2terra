local th = terralib.includec("/mnt/raid/torch/install/include/TH/TH.h")
assert(terralib.loadfile("wrap.t"))()

local C = terralib.includec("stdio.h")




local tensorTypes = {}
tensorTypes["DoubleTensor"] = true
tensorTypes["ByteTensor"] = true

local torchTypes = {}
torchTypes["DoubleTensor"] = &th.THDoubleTensor
torchTypes["ByteTensor"] = &th.THByteTensor
torchTypes["double"] = double
torchTypes["index"] = int32

local DoubleTensorStr="DoubleTensor"
local DoubleTensorType = &th.THDoubleTensor

local ByteTensorStr="ByteTensor"
local ByteTensorType = &th.THByteTensor

local realStr = "double"
local realType = double

local real = realStr
local Tensor = DoubleTensorStr

cwrapFun = require 'cwrapFun'
local cwrap = cwrapFun:init(Tensor)
cwrapFun:checkCwrap(cwrap)

function createFun(op,argTypes)
  
  
  --Takes in a cwrap table
  --checks if argTypes is viable for that cwrap table
  --returns isFit, cwrapMap
  --  isFit: true if that table works for argTypes
  --  crwapMap: a map from argTypes idxs to cwrap table idxs 
  local function checkFit(t)
    
    local function sameType(argTypeStr,t)
      if tensorTypes[argTypeStr] then
        return (argTypeStr==t.name)
      elseif argTypeStr=="number" then
        return (t.name=='index' or t.name=='double')
      else
        return false
      end
    end
    
    --TODO
    --Does not handle the case when it fits but there are more necessary args at the end of the table
    local function recCheckFit(argI,tI)
      if(argI > #argTypes) then
        return true, {}
      elseif(tI > #t) then 
        return false, nil
      end
      if(t[tI].creturned) then
        return recCheckFit(argI,tI+1)
      end
      if(sameType(argTypes[argI],t[tI])) then
        local boolRet, tab = recCheckFit(argI+1,tI+1) 
        if boolRet then
          table.insert(tab,tI)
          return true, tab
        end
      end
      if(t[tI].default) then
        return recCheckFit(argI,tI+1)
      else
        return false, nil
      end
    end
    return recCheckFit(1,1)
  end

  local function reverseTable(t)
    for i=1,math.floor(#t/2) do 
      t[i],t[#t-i+1]=t[#t-i+1],t[i]
    end
  end

  local cwrapOp = cwrap[op]
  local cName, cwrapMap, cwrapTab = nil,nil,nil
  for i = 1,#cwrapOp/2 do
    cwrapTab = cwrapOp[2*i]
    local isFit, tab = checkFit(cwrapTab)
    if isFit then
      cName = cwrapOp[2*i-1]
      cwrapMap = tab
      reverseTable(cwrapMap)
      break
    end
  end
  
  print(argTypes)
  assert(cName,"ERROR: pattern not found for",op,argTypes)
  
  
  --create tArgTypes
  local tArgTypes = {}
  for _,idx in ipairs(cwrapMap) do
    table.insert(tArgTypes,cwrapTab[idx].name)
  end

  --create cArgTypes
  local cArgTypes = {}
  for _,tab in ipairs(cwrapTab) do
    if not tab.creturned then
      table.insert(cArgTypes,tab.name)
    end
  end
  
  --create argMap
  local argMap = {}
  local argIdx = 1
  for i=1,#cwrapTab do
    if cwrapMap[argIdx]==i then
      argMap[i] = argIdx
      argIdx = argIdx+1
    end
  end

  --create returnMap
  local returnMap = {}
  local cRet = {}
  for i,t in ipairs(cwrapTab) do
    if(t.creturned) then
      assert(next(returnMap)==nil,"ERROR: both creturned and return")
      cRet.i,cRet.type = i, torchTypes[t.name]
    end
    if(t.returned) then
      assert(cRet.i==nil)
      table.insert(returnMap,i)
    end
  end

  --print("argTypes",argTypes)
  --print("tArgTypes",tArgTypes)
  --print("cwrapMap",cwrapMap)
  --print("cName",cName)
  --print("cArgTypes",cArgTypes)
  --print("argMap",argMap)
  --print("returnMap",returnMap)
  
  --cName is the c function name
  --cArgTypes is an array of the types of the c function
  --argMap is map between C arg list and passed in lua args
  --returnMap is a table with 1 containg first return Cvalue idx, 2 second return Cvalue idx and so on
  
  --local function createTerraFun(cName,cArgTypes,argMap,returnMap)
  local function createTerraFun()
    
    local function getType(typeStr)
      if torchTypes[typeStr]==nil then
        assert(false,"ERROR: bad argType! "..typeStr)
      end
      return torchTypes[typeStr]
    end
    
    local terraArgs = {}
    for i,tArgType in ipairs(tArgTypes) do
      table.insert(terraArgs,symbol(getType(tArgType),"arg"..i))
    end

    local cArgTable = {}
    local cArgRealTypes = {}
    for i,argType in ipairs(cArgTypes) do
      table.insert(cArgTable,symbol(getType(argType),"cArg"..i))
      table.insert(cArgRealTypes,getType(argType))
    end

    --TODO what is the actual default value??
    local function precallGen()
      local function getValue(i)
        if argMap[i]==nil then
          if cArgTypes[i]==DoubleTensorStr then
            return `th.THDoubleTensor_new()
          elseif cArgTypes[i]==ByteTensorStr then
            return `th.THByteTensor_new()
          else
            return `1
          end
        else
          if cArgTypes[i]=="index" then
            return `[terraArgs[argMap[i]]] - 1
          else
            return terraArgs[argMap[i]]
          end
        end
      end
      local precallTable = {}
      for i,cArgSym in ipairs(cArgTable) do
        precallTable[i] = quote 
          var [cArgSym] = [getValue(i)]
        end
      end
      return precallTable
    end

    local cRetType = {}
    if cRet.type then
      cRetType = {cRet.type}
    end
    
    local cCall = terralib.externfunction(cName,cArgRealTypes->cRetType)
    
    local function cCallGen()
      if cRet.i then
        cRet.sym = symbol(cRet.type,"cArg"..(cRet.i))
        return quote
          var [cRet.sym] = cCall([cArgTable])
          --C.printf("%d\n",[cRet.sym]) 
        end
      else
        return `cCall([cArgTable])
      end
    end
    
    local function returnGen()
      if cRet.sym then
        return cRet.sym
      end
      local returnList = {}
      for i,argNum in ipairs(returnMap) do
        table.insert(returnList,cArgTable[argNum])
      end
      return returnList
    end
    

    return terra ([terraArgs])
      [precallGen()]
      [cCallGen()]
      return [returnGen()]
    end
  end


  local terraFun = createTerraFun()
  terraFun:printpretty()

  -- a list of outputs.
  -- [1] is first return, [2] is second...
  -- contains a tuple of {cmd,idx} 
  --cmd : 1 cRet. just return terraRets[idx=1]
  --      2 new object. Wrap and return terraRets[idx]
  --      3 orig arg.   just return args[idx]
  local returnCmd = {}
  if cRet.i then
    table.insert(returnCmd,{1,1})
  else
    for i,retIdx in pairs(returnMap) do
      if argMap[retIdx] then --Return value is arg passed in
        table.insert(returnCmd,{3,argMap[retIdx]})
      else
        table.insert(returnCmd,{2,i,cArgTypes[retIdx]})
      end
    end
  end
  local TensorTypeMap = {}
    TensorTypeMap["DoubleTensor"] = "torch.DoubleTensor"
    TensorTypeMap["ByteTensor"] = "torch.ByteTensor"

  return function(...)
    local args = {...}
    local terraArgs = {}
    for i,arg in ipairs(args) do
      if tensorTypes[tArgTypes[i]] then
        table.insert(terraArgs,unwrapTorchObject(arg))
      else
        table.insert(terraArgs,arg)
      end
    end
    local terraRets = table.pack(terralib.unpackstruct(terraFun(table.unpack(terraArgs))))
    local rets = {}
    for i,tup in ipairs(returnCmd) do
      if(tup[1]==1) then
        table.insert(rets,terraRets[tup[2]])
      elseif(tup[1]==2) then
        table.insert(rets,wrapTorchObject(terraRets[tup[2]], TensorTypeMap[tup[3]]))
      elseif(tup[1]==3) then
        table.insert(rets,args[tup[2]])
      end
    end
    return table.unpack(rets)
  end
end
