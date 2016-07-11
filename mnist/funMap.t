assert(terralib.loadfile("wrap.t"))()
local types = assert(terralib.loadfile("types.t"))()
assert(terralib.loadfile("tWrap.t"))()

local C = terralib.includec("stdio.h")


cArgsList = require 'cArgsList'

function createFun(op,argTypes)
  
  --Takes in a cArgsList table
  --checks if argTypes is viable for that cArgsList table
  --returns isFit, cArgsListMap
  --  isFit: true if that table works for argTypes
  --  crwapMap: a map from argTypes idxs to cArgsList table idxs 
  local function checkFit(t)
    
    local function sameType(argTypeStr,t)
      if types.tensorTypes[argTypeStr] then
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

  local cArgsListOp = cArgsList.list[op]
  local cName, cArgsListMap, cArgsListTab = nil,nil,nil
  for i = 1,#cArgsListOp/2 do
    cArgsListTab = cArgsListOp[2*i]
    local isFit, tab = checkFit(cArgsListTab)
    if isFit then
      cName = cArgsListOp[2*i-1]
      cArgsListMap = tab
      reverseTable(cArgsListMap)
      break
    end
  end
  assert(cName,"ERROR: pattern not found for",op,argTypes)

  local wrap = tWrap.new(cName,cArgsListTab,cArgsListMap)

  local terraFun = wrap:tWrapFun()
  
  local returnCmd = wrap:getReturnCmd()
  
  terraFun:printpretty()

  return function(...)
    local args = {...}
    local terraArgs = {}
    for i,arg in ipairs(args) do
      if types.tensorTypes[wrap.tArgTypes[i]] then
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
        table.insert(rets,wrapTorchObject(terraRets[tup[2]], tup[3]))
      elseif(tup[1]==3) then
        table.insert(rets,args[tup[2]])
      end
    end
    return table.unpack(rets)
  end
end
