local types = require 'types'
require "torchTerraWrap"
require "terraGen"

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

local Cstdio = terralib.includec("stdio.h")
local Cstdlib = terralib.includec("stdlib.h")
local rlocalsSave = nil
local luaFunCache = {}
local terraFunCache = {}
local callSeq = {}
local constantStorageCache = {}

--Convenience functions
c.Expr.torchType = function(self)
  return "torch."..tostring(self.type)
end
c.Expr.terraType = function(self)
  return types.getType(self:torchType())
end

--TODO this is all hacky. Is there a asdl function that does this?
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

--Add the basic binary operators
for _,op in pairs({'add','sub','mul','div'}) do
  c.Expr["__"..op] = function(lhs,rhs)
    local lhsExpr,rhsExpr = lhs, rhs
    if type(lhsExpr)=="number" then
      lhsExpr = c.Constant(lhsExpr,c.Number)
    end
    assert(lhsExpr.type==c.Number,"Error: Only supports scalar operations")
    
    if type(rhsExpr)=="number" then
      rhsExpr = c.Constant(rhsExpr,c.Number)
    end
    assert(rhsExpr.type==c.Number,"Error: Only supports scalar operations")
    local terraOp = terra(lhs : double, rhs : double)
      escape 
        if op=='add' then
          emit quote return lhs + rhs end
        elseif op=='sub' then
          emit quote return lhs - rhs end
        elseif op=='mul' then
          emit quote return lhs * rhs end
        elseif op=='div' then
          emit quote return lhs / rhs end
        end
      end
    end
    local retExpr = c.CallWrap(terraOp,"terra"..op,List({lhsExpr,rhsExpr}),c.Number)
    table.insert(callSeq,retExpr)
    return retExpr
  end
end

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
local rlStructTypes = {}

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
    
    --Create struct holding all rlocals
    local rlStruct = terralib.types.newstruct("rlStructName")
    for idx,type in pairs(rlStructTypes) do
      local name = 'rl'..idx
      rlStruct.entries:insert({field=name,type=type})
    end

    local symTable = {}
    local rlSymbol = symbol(&rlStruct,"rlSym")
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
        assert(rlStructTypes[expr.position],"Did not find rlStruct"..expr.position)
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
      local callArgs = {}
      for i,argExpr in ipairs(callExpr.args) do
        local sym = getSymbol(argExpr)
        assert(sym)
        table.insert(callArgs,sym)
      end
      local name = callExpr:getSymName()
      assert(not symTable[name],"Found name when should not have")
      symTable[name] = symbol(callExpr:terraType(),name)
      table.insert(terraCodeBody,quote var [symTable[name]] = [callExpr.terrafun]([callArgs]) end)
    end
    
    --create Terra arg list
    --Terra arg list ordering (rlocal, constants..., ordered gradient params, other params)
    --gpStruct
    local offset = 1
    local terraArgSymbols = {rlSymbol}
    for _,expr in pairs(constantStorageCache) do
      assert(not terraArgSymbols[expr.idx+offset])
      terraArgSymbols[expr.idx+offset] = getSymbol(expr)
    end
    offset = #terraArgSymbols
    
    --Create ordered flattened gpExpr list
    local gpExprList = {}
    local function genGpList(t)
      if t._isExpr then
        table.insert(gpExprList,t)
      else
        assert(t._sorted)
        for _,k in ipairs(t._sorted) do
          genGpList(t[k])
        end
      end
    end
    genGpList(gpExpr)
    
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
    
    --Create the actual terra function
    local terraFun = terra([terraArgSymbols])
      [terraCodeBody]
      return [retSymbols]
    end
    
    -- Connstruct and load the rlocals struct
    terra constructRlStruct()
      var r : &rlStruct = [&rlStruct](Cstdlib.malloc(sizeof(rlStruct)))
      escape
        for i,_ in pairs(rlStructTypes) do
          local name = 'rl'..i
          local rType = types.getType(tostring(torch.type(rlocalsSave[i])))
          emit quote r.[name] = [unwrapTorchObject(rlocalsSave[i],rType)] end
        end
      end
      return r
    end
    local tArgRlObj = constructRlStruct()
    
    --Load the constants
    local tArgC = {}
    for storage,expr in pairs(constantStorageCache) do
      tArgC[expr.idx] = unwrapTorchObject(storage)
    end
    
    --Prep gpRet with correct nested structure
    local gpRet = {}
    local function preNest(ref,new)
      if type(ref)~="table" then
        return
      else
        for k,v in pairs(ref) do
          new[k] = {}
          preNest(v,new[k])
        end
      end
    end
    preNest(gps,gpRet)

    local function getRet(tRets,i)    
      if retMap[i].rlocal then
        return rlocalsSave[retMap[i].rlocal]
      elseif retMap[i].torchType then
        return wrapTorchObject(tRets[i],retMap[i].torchType)
      else 
        return tRets[i]
      end
    end

    --Final lua function which parses and passes the args into the terra function
    -- This should be simple as it is on the critical path
    return function(gp,...)
      local tArgs = {}
      --rlocal struct
      table.insert(tArgs,tArgRlObj)
      
      --Constants
      for _,tArg in ipairs(tArgC) do
        table.insert(tArgs,tArg)
      end
      
      --Ordered gradient parameters (gp)
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
      
      --unpack the terra reteurns into the correct format
      for i,expr in ipairs(gpExprList) do
        local r = gpRet
        for j,k in ipairs(expr.path) do
          if(j==#expr.path) then
            r[k] = getRet(tRets,i)
          else
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
  end --End createLuaFun
  
  --Flatten a table to a string of types
  local function tab2str(t)
    tType = type(t)
    if tType=='userdata' then
      return torch.type(t)
    elseif tType~='table' then
      return tType
    end
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

  --Create function only if it has not been created before
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
  
  --TODO Hack because autograd defines the functions after this code is generated
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
    local argExps = {}
    for i,arg in ipairs(args) do
      if(type(arg)=="number") then
        arg = c.Constant(arg,c.Number)
      elseif(torch.isStorage(arg)) then
        if not constantStorageCache[arg] then
          constantStorageCache[arg] = c.ConstantStorageWrap(arg,c.LongStorage)
        end
        arg = constantStorageCache[arg]
      elseif(arg:isclass(c.RLocal)) then
        rlStructTypes[arg.position] = arg:terraType()
      end 
      table.insert(argExps,arg)
      table.insert(argTypes,"torch."..tostring(arg.type))
    end
    local argTypeStr = table.concat(argTypes,"__")
    if terraFunCache[fun] == nil then
      terraFunCache[fun] = {}
    end
    if terraFunCache[fun][argTypeStr] == nil then
      terraFunCache[fun][argTypeStr] = createTerraObj(fun,argTypes,name)
    end
    local funObj = terraFunCache[fun][argTypeStr]
    local retExpr = c.CallWrap(funObj.fun, name,List(argExps),c[funObj.returnType[1]])
    table.insert(callSeq,retExpr)
    return retExpr
  end
end
