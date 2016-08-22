local th = terralib.includec("/mnt/raid/torch/install/include/TH/TH.h")
local types, cArg = assert(terralib.loadfile("types.t"))()
tWrap = {}

function tWrap.new(cFun,cArgsListTab,cArgsListMap)
  local self = {cFun=cFun,cArgsTab=cArgsListTab,cArgsMap=cArgsListMap}
  setmetatable(self, {__index=tWrap})
  
  --cFun is the c function name
  --cArgs is an array of cArg objects
  --argMap is map between C arg list and passed in lua args
  --returnMap is a table with 1 containg first return Cvalue idx, 2 second return Cvalue idx and so on
  
 
  --create tArgTypes
  local tArgTypes = {}
  for _,idx in ipairs(self.cArgsMap) do
    table.insert(tArgTypes,self.cArgsTab[idx].name)
  end
 
  --create tArgSymbols
  local tArgSymbols = {}
  for i,tArgType in ipairs(tArgTypes) do
    table.insert(tArgSymbols,symbol(types.getType(tArgType),"tArg"..i))
  end
 
   --create argMap
  local argMap = {}
  local argIdx = 1
  for i=1,#self.cArgsTab do
    if self.cArgsMap[argIdx]==i then
      argMap[i] = argIdx
      argIdx = argIdx+1
    end
  end
 
  --create cArgs
  local cArgs = {}
  local cRet = nil
  for i,tab in ipairs(self.cArgsTab) do
    if i==#self.cArgsTab and tab.creturned then
      cRet = cArg.new(i,argMap[i],tab)
    else
      table.insert(cArgs,cArg.new(i,argMap[i],tab))
    end
  end
  --let each cArg have access to the array of cArgs
  for i,cArg in ipairs(cArgs) do
    cArg.cArgs = cArgs
  end

  --create returnMap
  local returnMap = {}
  for i,t in ipairs(self.cArgsTab) do
    if(t.creturned) then
      assert(next(returnMap)==nil,"ERROR: both creturned and return")
    end
    if(t.returned) then
      table.insert(returnMap,i)
    end
  end
  
  local cArgSymbols = {}
  local cArgTypes = {}
  for i,cArg in ipairs(cArgs) do
    table.insert(cArgSymbols,cArg.symbol)
    table.insert(cArgTypes,types.getType(cArg.name))
  end

  self.tArgTypes = tArgTypes
  self.tArgSymbols = tArgSymbols
  self.argMap = argMap
  self.cArgs = cArgs
  self.returnMap = returnMap
  self.cRet = cRet
  self.cArgSymbols = cArgSymbols
  self.cArgTypes = cArgTypes
  --print("tArgTypes",tArgTypes)
  --print("self.cArgsMap",self.cArgsMap)
  --print("cFun",self.cFun)
  --print("cArgTypes",cArgTypes)
  --print("argMap",argMap)
  --print("returnMap",returnMap)

  return self
end

function tWrap:tWrapFun()
  
  
  local cRet = self.cRet
  local cRetType = {}
  if cRet then
    cRetType = {types.getType(cRet.name)}
  end
  
  local cCall
  if(type(self.cFun)=="string") then
    cCall = terralib.externfunction(self.cFun,self.cArgTypes->cRetType)
  else
    cCall = self.cFun
  end

  local function cCallGen()
    if cRet then
      return quote
        var [cRet.symbol] = cCall([self.cArgSymbols])
      end
    else
      return `cCall([self.cArgSymbols])
    end
  end
 
  local function returnGen()
    if cRet then
      return cRet.symbol
    end
    local returnList = {}
    for i,argNum in ipairs(self.returnMap) do
      table.insert(returnList,self.cArgSymbols[argNum])
    end
    return returnList
  end

  return terra ([self.tArgSymbols])
    escape 
      for i,cArg in ipairs(self.cArgs) do
        emit quote [cArg:init(self.tArgSymbols)] end
      end
      for i,cArg in ipairs(self.cArgs) do
        emit quote [cArg:precall()] end
      end
    end
    [cCallGen()]
    return [returnGen()]
  end

end

function tWrap:getReturnCmd()
  -- a list of outputs.
  -- [1] is first return, [2] is second...
  -- contains a tuple of {cmd,idx} 
  --cmd : 1 cRet. just return terraRets[idx=1]
  --      2 new object. Wrap and return terraRets[idx]
  --      3 orig arg.   just return args[idx]
  returnCmd = {}
  if self.cRet then
    if(types.tensorTypes[self.cRet.name]) then
        table.insert(returnCmd,{2,1,self.cRet.name})
    else
      table.insert(returnCmd,{1,1})
    end
  else
    for i,retIdx in pairs(self.returnMap) do
      if self.argMap[retIdx] then --Return value is arg passed in
        table.insert(returnCmd,{3,self.argMap[retIdx]})
      else
        table.insert(returnCmd,{2,i,self.cArgs[retIdx].name})
      end
    end
  end
  return returnCmd
  
end

function tWrap:getReturnType()
  returnTypes = {}
  if self.cRet then
    table.insert(returnTypes,self.cRet.name)
  else
    for i,retIdx in pairs(self.returnMap) do
      if self.argMap[retIdx] then --Return value is arg passed in
        table.insert(returnTypes,self.cArgs[self.argMap[retIdx]].name)
      else
        table.insert(returnTypes,self.cArgs[retIdx].name)
      end
    end
  end
  for i,str in ipairs(returnTypes) do
    if(str:match("torch")) then
      returnTypes[i] = str:match("torch.(%S+)")
    elseif(str=="double") then
      returnTypes[i] = "Number"
    else
      assert(false,"BAD type!! " .. str)
    end
  end
  return returnTypes
end


return tWrap
