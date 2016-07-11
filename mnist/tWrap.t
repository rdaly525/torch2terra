local th = terralib.includec("/mnt/raid/torch/install/include/TH/TH.h")
local types = assert(terralib.loadfile("types.t"))()

tWrap = {}

function tWrap.new(cName,cArgsListTab,cArgsListMap)
  local self = {cName=cName,cArgsTab=cArgsListTab,cArgsMap=cArgsListMap}
  setmetatable(self, {__index=tWrap})
  return self
end

function tWrap:tWrapFun()
  
  --create tArgTypes
  local tArgTypes = {}
  for _,idx in ipairs(self.cArgsMap) do
    table.insert(tArgTypes,self.cArgsTab[idx].name)
  end

  --create cArgTypes
  local cArgTypes = {}
  for _,tab in ipairs(self.cArgsTab) do
    if not tab.creturned then
      table.insert(cArgTypes,tab.name)
    end
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

  --create returnMap
  local returnMap = {}
  local cRet = {}
  for i,t in ipairs(self.cArgsTab) do
    if(t.creturned) then
      assert(next(returnMap)==nil,"ERROR: both creturned and return")
      cRet.i,cRet.type = i, types.torchTypes[t.name]
    end
    if(t.returned) then
      assert(cRet.i==nil)
      table.insert(returnMap,i)
    end
  end

  --print("tArgTypes",tArgTypes)
  --print("self.cArgsMap",self.cArgsMap)
  --print("cName",self.cName)
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
      if types.torchTypes[typeStr]==nil then
        assert(false,"ERROR: bad argType! "..typeStr)
      end
      return types.torchTypes[typeStr]
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
          if cArgTypes[i]==types.DoubleTensorStr then
            return `th.THDoubleTensor_new()
          elseif cArgTypes[i]==types.ByteTensorStr then
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
    
    local cCall = terralib.externfunction(self.cName,cArgRealTypes->cRetType)
    
    local function cCallGen()
      if cRet.i then
        cRet.sym = symbol(cRet.type,"cArg"..(cRet.i))
        return quote
          var [cRet.sym] = cCall([cArgTable])
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
  
  self.returnCmd = {}
  if cRet.i then
    table.insert(self.returnCmd,{1,1})
  else
    for i,retIdx in pairs(returnMap) do
      if argMap[retIdx] then --Return value is arg passed in
        table.insert(self.returnCmd,{3,argMap[retIdx]})
      else
        table.insert(self.returnCmd,{2,i,cArgTypes[retIdx]})
      end
    end
  end
  
  --TODO hacky
  self.tArgTypes = tArgTypes

  local terraFun = createTerraFun()
  return terraFun

end

function tWrap:getReturnCmd()
  -- a list of outputs.
  -- [1] is first return, [2] is second...
  -- contains a tuple of {cmd,idx} 
  --cmd : 1 cRet. just return terraRets[idx=1]
  --      2 new object. Wrap and return terraRets[idx]
  --      3 orig arg.   just return args[idx]
  return self.returnCmd
end

return tWrap
