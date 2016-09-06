local types = require 'terragen.types'
require "terragen.torchTerraWrap"
require "terragen.terraGen"

local asdl = require("asdl")
local List = asdl.List
local c = asdl.NewContext()


c:Extern("terrafunction",function(t) return terralib.isfunction(t) end)
c:Extern("longstorage",function(t) return torch.type(t)=="torch.LongStorage" end)
c:Define [[
Type = DoubleTensor
    | ByteTensor
    | LongTensor
    | LongStorage
    | Number

Expr = Param(number position, string name, any* path)
    | Call(terrafunction terrafun, string funname, Expr* args)
    | RLocal(number position)
    | Constant(number value)
    | ConstantStorage(longstorage storage)
    attributes(Type type)
]]

--generalize dealing with constants (LongStorage)
--Better way to deal with nested parameters (integrate with asdl?)

local rlocalsSave = nil
local luaFunCache = {}
local terraFunCache = {}
local callSeq = {}
local constantStorageCache = {}

c.Expr.torchType = function(self)
  return "torch."..tostring(self.type)
end
c.Expr.terraType = function(self)
  return types.getType(self:torchType())
end

--TODO this is all hacky
c.Expr.isclass = function(self,c)
  return getmetatable(self)==c
end
c.Expr._isExpr =true
c.Expr.getSymName = function(self)
  return self.name
end

c.RLocal.getSymName = function(self) return 'rl'..self.position end
c.ConstantStorage.getSymName = function(self) return 'cs'..self.idx end
c.Call.getSymName = function(self) return 'call_'..self.funname..self.idx end


c.ConstantStorage.idx = 1
c.ConstantStorageWrap = function(...)
  local pos = c.ConstantStorage.idx
  local obj = c.ConstantStorage(...)
  obj.idx = pos
  c.ConstantStorage.idx = pos+1
  return obj
end

c.Call.idx = 1
c.CallWrap = function(...)
  local pos = c.Call.idx
  local obj = c.Call(...)
  obj.idx = pos
  c.Call.idx = pos+1
  return obj
end

c.Expr.__div = function(lhs,rhs)
  assert(lhs.type==c.Number and type(rhs)=="number","Only supports scalar div")
  local rhsExpr = c.Constant(rhs,c.Number)
  local terraDiv = terra(lhs : double, rhs : double)
    return lhs / rhs
  end
  local retExpr = c.CallWrap(terraDiv,"terradiv",List({lhs,rhsExpr}),lhs.type)
  table.insert(callSeq,retExpr)
  return retExpr
end

--function createParam(position)
--  local function createSubParam(position,path)
--    function indexFun(t,k)
--      local newPath = {}
--      for _,v in ipairs(t.path) do
--        table.insert(newPath,v)
--      end
--      table.insert(newPath,k)
--      t[k] = createSubParam(t.position,newPath)
--      return t[k]
--    end
--    local obj = {position=position,path=path,nestedparam=true}
--    setmetatable(obj,{__index=indexFun})
--    return obj
--  end
--  local param = createSubParam(position,{})
--  --local param = createSubParam(position,{})
--  return param
--end

local function str2exprType(str)
  if(str:match("torch")) then
    str = str:match("torch.(%S+)")
  end
  assert(c[str],str.." is not a expression type!")
  return c[str]
end

function rlocalsWrap(rlocals)
  rlocalsSave = rlocals
  local rLocalsExps = {}
  for i,rlocal in ipairs(rlocals) do 
    local exprType = str2exprType(torch.type(rlocal))
    table.insert(rLocalsExps,c.RLocal(i,exprType))
  end
  return rLocalsExps
end

--Maps names to types
local rlStructIdxs = {}

function compileTrace(gradFun,rlocalsWrap)

  local function createParam(t,idx)
    local function copy(t)
      n = {}
      for k,v in pairs(t) do
        n[k] = v
      end
      return n
    end
    local function typeTable(t,keys,name)
      tType = type(t)
      if tType~='table' then
        if tType=='userdata' then
          tType = torch.type(t)
        end
        return c.Param(idx,name.."_"..table.concat(keys,"_"),List(keys),str2exprType(tType))
      end
      local sorted = {}
      local newTab = {}
      for k,v in pairs(t) do
        table.insert(sorted,k)
        nkeys = copy(keys)
        table.insert(nkeys,k)
        newTab[k] = typeTable(v,nkeys,name)
      end
      table.sort(sorted)
      newTab._sorted=sorted
      return newTab
    end
    return typeTable(t,{},"p"..idx)
  end
  --gp = grad parameter
  --op = other parameter
  local function createLuaFun(gps,...)  
    local gpExpr = createParam(gps,1)
    local opExprs = {}
    for i,opExpr in ipairs({...}) do
      table.insert(opExprs,createParam(opExpr,i+1))
    end
    local gradFunRet = {gradFun(gpExpr,table.unpack(opExprs))}
    
    --local function genStruct(t,name)
    --  if t._isExpr then
    --    return t:terraType()
    --  end
    --  assert(t._sorted)
    --  local newStruct = terralib.types.newstruct(name)
    --  for _,k in ipairs(t._sorted) do
    --    local type = genStruct(t[k],name.."_"..k)
    --    newStruct.entries:insert({field=k,type=type})
    --  end
    --  return newStruct
    --end
    --local gpStruct = genStruct(gpExpr,"gpStruct")

    local rlStruct = terralib.types.newstruct("rlStructName")
    for idx,type in pairs(rlStructIdxs) do
      local name = 'rl'..idx
      rlStruct.entries:insert({field=name,type=type})
    end

    local symTable = {}
    local rlSymbol = symbol(rlStruct,"rlSym")
    local function getSymbol(expr)
      local name = expr:getSymName()
      if(expr:isclass(c.Constant)) then
        return `[expr.value]
      elseif(expr:isclass(c.Param)) then
        if not symTable[name] then
          symTable[name] = symbol(expr:terraType(),name)
        end
        return symTable[name]
      elseif(expr:isclass(c.RLocal)) then
        assert(rlStructIdxs[expr.position],"Did not find rlStruct"..expr.position)
        return `[rlSymbol].[name]
      elseif(expr:isclass(c.ConstantStorage)) then
        if not symTable[name] then
          symTable[name] = symbol(expr:terraType(),name)
        end
        return symTable[name]
      elseif(expr:isclass(c.Call)) then
        assert(symTable[name],"Should have found call: "..name)
        return symTable[name]
      end
    end
    
    --Loop through all the function calls to generate terra code
    local terraCodeBody = {}
    for i,callExpr in ipairs(callSeq) do
      local name = callExpr:getSymName()
      local callArgs = {}
      for i,argExpr in ipairs(callExpr.args) do
        local sym = getSymbol(argExpr)
        assert(sym)
        table.insert(callArgs,sym)
      end
      assert(not symTable[name],"Found name when should not have")
      symTable[name] = symbol(callExpr:terraType(),name)
      table.insert(terraCodeBody,quote var [symTable[name]] = [callExpr.terrafun]([callArgs]) end)
    end
    
    --create Terra arg list
    --Terra arg list ordering (rlocal, constants..., gradient param struct, other params)
    --gpStruct
    local offset = 1
    local terraArgSymbols = {rlSymbol}
    for _,expr in pairs(constantStorageCache) do
      assert(not terraArgSymbols[expr.idx+offset])
      terraArgSymbols[expr.idx+offset] = getSymbol(expr)
    end
    offset = #terraArgSymbols
    
    local gpExprList = {}
    local function genList(t)
      if t._isExpr then
        table.insert(gpExprList,t)
      else
        assert(t._sorted)
        for _,k in ipairs(t._sorted) do
          genList(t[k])
        end
      end
    end
    genList(gpExpr)
    
    --local gpArgSymbol = symbol(gpStruct,"gpStruct"))
    --table.insert(terraArgSymbols,gpArgSymbol)

    for _,expr in ipairs(gpExprList) do
      table.insert(terraArgSymbols,symTable[expr:getSymName()])
    end

    for _,expr in ipairs(opExprs) do
      table.insert(terraArgSymbols,symTable[expr:getSymName()])
    end
    
    --Determine and order the terra returns
    --Also create a returnmap that lua function can interpret
    local function getRetMap(expr)
      return {
        rlocal = expr:isclass(c.RLocal) and expr.position,
        torchType= expr.type~=c.Number and expr:torchType()
      }
    end
    local retSymbols = {}
    local retMap = {}
    for _,expr in ipairs(gpExprList) do
      local rExpr = gradFunRet[1]
      for _,k in ipairs(expr.path) do
        rExpr = rExpr[k]
      end
      table.insert(retSymbols,getSymbol(rExpr))
      table.insert(retMap,getRetMap(rExpr))
    end
    for i=2,#gradFunRet do
      local rExpr = gradFunRet[i]
      table.insert(retSymbols,getSymbol(rExpr))
      table.insert(retMap,getRetMap(rExpr))
    end
    
    local terraFun = terra([terraArgSymbols])
      [terraCodeBody]
      return [retSymbols]
    end
    --terraFun:printpretty()
    
    terra constructRlStruct()
      var r : rlStruct
      escape
        for i,_ in pairs(rlStructIdxs) do
          local name = 'rl'..i
          local rType = types.getType(tostring(torch.type(rlocalsSave[i])))
          emit quote r.[name] = [unwrapTorchObject(rlocalsSave[i],rType)] end
        end
      end
      return r
    end
    local tArgRlObj = constructRlStruct()
    local tArgC = {}
    for storage,expr in pairs(constantStorageCache) do
      tArgC[expr.idx] = unwrapTorchObject(storage)
    end
    
    --Prep gpRet with correct nested structure
    local function getRet(tRets,i)    
      if retMap[i].rlocal then
        return rlocalsSave[retMap[i].rlocal]
      elseif retMap[i].torchType then
        return wrapTorchObject(tRets[i],retMap[i].torchType)
      else 
        return tRets[i]
      end
    end

    return function(gp,...)
      local tArgs = {}
      --rlocal struct
      table.insert(tArgs,tArgRlObj)
      
      --Constants
      for _,tArg in ipairs(tArgC) do
        table.insert(tArgs,tArg)
      end
      
      --Ordered gradient parameters (p1)
      for _,expr in ipairs(gpExprList) do
        local p = gp
        for _,k in ipairs(expr.path) do
          p = p[k]
        end
        table.insert(tArgs,unwrapTorchObject(p))
      end
      
      --Other parameters
      for _,p in ipairs({...}) do
        table.insert(tArgs,unwrapTorchObject(p))
      end
      
      --Call the terraFunction
      local tRets = table.pack(terralib.unpackstruct(terraFun(table.unpack(tArgs))))
      
      local gpRet = {}
      for i,expr in ipairs(gpExprList) do
        local r = gpRet
        for j,k in ipairs(expr.path) do
          if(j==#expr.path) then
            r[k] = getRet(tRets,i)
          else
            r[k] = {}
            r = r[k]
          end
        end
      end
      local oRets = {}
      for i=#gpExprList+1,#tRets do
        table.insert(oRets,getRet(tRets,i))
      end
      return gpRet,table.unpack(oRets)
    end
  end
  
  --Flatten a table to a string of types
  local function tab2str(t)
    tType = type(t)
    if tType=='userdata' then
      return torch.type(t)
    elseif tType~='table' then
      return tType
    end
    assert(tType=='table')
    sorted = {}
    for k,_ in pairs(t) do
      table.insert(sorted,k)
    end
    table.sort(sorted)
    str = "["
    for _,k in ipairs(sorted) do
      str = str .. "_"..k.."_"..tab2str(t[k])
    end
    return str.."]"
  end

  --Create function only if it has not been cerated before
  return function(gparams,...)
    local pStr = tab2str(gparams)
    for _,p in ipairs({...}) do
      pStr = pStr .."_"..tab2str(p)
    end
    if not luaFunCache[pStr] then
      luaFunCache[pStr] = createLuaFun(gparams,...)
    end
    return luaFunCache[pStr](gparams,...)
  end
end



function torchWrap(name,fun)
  
  --TODO Hack because autograd definies the functions later
  --Just using string as cache instead of fun
  if(name=="torch.t") then
    fun = "torch.t"
  elseif(name=="util.typeAsInPlace") then
    fun = "util.typeAsInPlace"
  elseif(name=="util.fillInPlace") then
    fun = "util.fillInPlace"
  end

  return function(...) 
    local args = {...}
    local argTypes = {}
    argExps = {}
    for i,arg in ipairs(args) do
      if(type(arg)=="number") then
        arg = c.Constant(arg,c.Number)
      elseif(torch.isStorage(arg)) then
        if not constantStorageCache[arg] then
          constantStorageCache[arg] = c.ConstantStorageWrap(arg,c.LongStorage)
        end
        arg = constantStorageCache[arg]
      elseif(arg:isclass(c.RLocal)) then
        rlStructIdxs[arg.position] = arg:terraType()
      end 
      table.insert(argExps,arg)
      table.insert(argTypes,"torch."..tostring(arg.type))
    end
    local argTypeStr = table.concat(argTypes,"__")
    if terraFunCache[fun] == nil then
      terraFunCache[fun] = {}
    end
    if terraFunCache[fun][argTypeStr] == nil then
      terraFunCache[fun][argTypeStr] = createTerraObj(fun,argTypes)
    end
    local funObj = terraFunCache[fun][argTypeStr]
    local retExpr = c.CallWrap(funObj.fun, name,List(argExps),c[funObj.returnType[1]])
    table.insert(callSeq,retExpr)
    return retExpr
  end
end
