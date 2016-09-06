local types = require "terragen.types"
local cArg = {types={}}

function cArg.new(i,ti,tab)
  local self = {i=i,ti=ti,name=tab.name,default=tab.default,returned=tab.returned,creturned=tab.creturned}
  assert(cArg.types[tab.name],"Bad Name!" .. tab.name)
  self.symbol =  symbol(types.torchTypes[tab.name],"cArg"..i)
  
  if(tab.precall) then
    self.precall = tab.precall
  end
  if(tab.defaultGen) then
    self.defaultGen = tab.defaultGen
  end

  setmetatable(self, {__index=cArg.types[tab.name]})
  return self
end

function cArg:init(tArgSymbols)
  if self.ti then 
    return quote
      var [self.symbol] = [tArgSymbols[self.ti]]
    end
  else
    assert(self.default,"Missing default!")
    return quote
      var [self.symbol] = [self:defaultGen()]
    end
  end
end

function cArg:precall()
  return quote
  end
end

function cArg:default(d)
  if (type(d)=="function") then
    return d(self)
  end
  assert(false,"No default!!")
end

function newType(typeStr,funTab)
  cArg.types[typeStr] = {__metatable = true}
  setmetatable(cArg.types[typeStr],{__index=cArg})
  for name,f in pairs(funTab) do
    cArg.types[typeStr][name] = f
  end
end


local typeStrs = {"DoubleTensor","LongTensor","ByteTensor","LongStorage"}

for _,t in ipairs(typeStrs) do
  newFunStr = "TH"..t.."_new"
  retType = types.torchTypes["torch."..t]
  local newFun = terralib.externfunction(newFunStr,{}->retType)
   
  newType("torch."..t,
    {
      defaultGen = function(self)
        if(type(self.default)=="number") then
          return `[self.cArgs[self.default].symbol]
        else
          return `newFun()
        end
      end
    }
  )
end

for _,t in ipairs({"double"}) do
  newType(t,
    {
      defaultGen = function(self)
        return `[self.default]
      end
    }
  )
end

newType("index",
  {
    defaultGen = function(self)
      return `[self.default]
    end,
    precall = function(self)
      local ret = quote
        [self.symbol] = [self.symbol] - 1
      end
      return ret
    end
  }
)

return cArg
