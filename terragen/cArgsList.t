local types = require "terragen.types"

local cArgsList = {}

function cArgsList:checkCArgs()
  for op, v in pairs(cArgsList.list) do
    assert(#v % 2 == 0)
    for i,args in ipairs(v) do
      assert((i%2==1 and (type(args)=="string" or terralib.type(args)=="terrafunction")) or (i%2==0 and type(args)=="table"),"BAD "..i.." "..type(args))
    end
    --TODO
    --Verify that creturned is always the last arg if it is there
    --for i=1,#v/2 do
    --  for ci,t in ipairs
    --end
  end
end


local reals = {ByteTensor='unsigned char',
             CharTensor='char',
             ShortTensor='short',
             IntTensor='int',
             LongTensor='long',
             FloatTensor='float',
             DoubleTensor='double'}

local accreals = {ByteTensor='long',
             CharTensor='long',
             ShortTensor='long',
             IntTensor='long',
             LongTensor='long',
             FloatTensor='double',
             DoubleTensor='double'}



function cArgsList:create(Tensor)
  
  local list = {}
  local TensorShort = Tensor:match("torch.(%S+)")
  local TensorShortShort = TensorShort:match("(%S+)Tensor")
  local function cname(name)
    return string.format('TH%s_%s', TensorShort,name)
  end
  local real = reals[TensorShort]
  local accreal = accreals[TensorShort]
  
  local tType = types.torchTypes[Tensor]
  local byteTensorType = types.torchTypes["torch.ByteTensor"] 
  local longStorageType = types.torchTypes["torch.LongStorage"]

  list[torch.add] = {
          cname("add"),
          {{name=Tensor, default=true, returned=true, method={default='nil'}},
           {name=Tensor, method={default=1}},
           {name=real}},
          cname("cadd"),
          {{name=Tensor, default=true, returned=true, method={default='nil'}},
           {name=Tensor, method={default=1}},
           {name=real, default=1},
           {name=Tensor}}}
  list[torch.neg] = {
       cname("neg"),
       {{name=Tensor, default=true, returned=true, method={default='nil'}},
        {name=Tensor, method={default=1}}}}
  list[torch.log] = {
       cname("log"),
       {{name=Tensor, default=true, returned=true, method={default='nil'}},
        {name=Tensor, method={default=1}}},
       "log",
       {{name=real},
        {name=real, creturned=true}}}
  list[torch.tanh] = {
       cname("tanh"),
       {{name=Tensor, default=true, returned=true, method={default='nil'}},
        {name=Tensor, method={default=1}}},
       "tanh",
       {{name=real},
        {name=real, creturned=true}}}
  list[torch.cdiv] = {
        cname("cdiv"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=Tensor}}}
  list[torch.cmul] = {
        cname("cmul"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=Tensor}}}
  list[torch.pow] = {
         cname("pow"),
         {{name=Tensor, default=true, returned=true, method={default='nil'}},
          {name=Tensor, method={default=1}},
          {name=real}},
         cname("tpow"),
         {{name=Tensor, default=true, returned=true, method={default='nil'}},
          {name=real},
          {name=Tensor, method={default=1}}},
         "pow",
         {{name=real},
          {name=real},
          {name=real, creturned=true}}}
  list[torch.exp] = {
        cname("exp"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}}},
        "exp",
        {{name=real},
         {name=real, creturned=true}}}
  local name = "eq"
  list[torch.eq] = {
         cname(name .. 'Value'),
         {{name='torch.ByteTensor',default=true, returned=true},
          {name=Tensor},
          {name=real}},
         cname(name .. 'ValueT'),
         {{name=Tensor, returned=true},
          {name=Tensor},
          {name=real}},
         cname(name .. 'Tensor'),
         {{name='torch.ByteTensor',default=true, returned=true},
          {name=Tensor},
          {name=Tensor}},
         cname(name .. 'TensorT'),
         {{name=Tensor, returned=true},
          {name=Tensor},
          {name=Tensor}}}
  name = nil
  list[torch.sum] = {
          cname("sumall"),
          {{name=Tensor},
           {name=accreal, creturned=true}},
          cname("sum"),
          {{name=Tensor, default=true, returned=true},
           {name=Tensor},
           {name="index"}} }
  
  name = "max"
  list[torch.max] = {
           cname(name .. "all"),
           {{name=Tensor},
            {name=real, creturned=true}},
           cname(name),
           {{name=Tensor, default=true, returned=true},
            {name="torch.LongTensor", default=true, noreadadd=true},
            {name=Tensor},
            {name="index"}},
           cname(name),
           {{name=Tensor, default=true, returned=true},
            {name="torch.LongTensor", default=true, returned=true, noreadadd=true},
            {name=Tensor},
            {name="index"}} }
    list[torch.mm] = {
      cname("addmm"),
      {
        {name=Tensor, default=true, returned=true, method={default='nil'},
          precall=function(self)
            local resize2dFun = terralib.externfunction("TH"..TensorShort.."_resize2d"      ,{tType,int32,int32}->{})
            
            local zeroFun = terralib.externfunction("TH"..TensorShort.."_zero",{tType}->{})
            return quote
              resize2dFun([self.symbol],[self.cArgs[5].symbol].size[0],[self.cArgs[6].symbol].size[1])
              zeroFun([self.symbol])
            end
          end
        },
        {name=real, default=0, invisible=true},
        {name=Tensor, default=1, invisible=true},
        {name=real, default=1, invisible=true},
        {name=Tensor, dim=2},
        {name=Tensor, dim=2} 
      }
    }
    --torch.t is not yet defined because it is called in support.lua
    --So I just map the function name to the function string
    
    list["torch.t"] = {
      cname("transpose"),
      {
        {name=Tensor, default=true, returned=true, dim=2},
        {name=Tensor, dim=2},
        {name='index', default=1, invisible=true},
        {name='index', default=2, invisible=true}
      }
    }
    local newWithTensorFun = terralib.externfunction(cname("newWithTensor"),{tType}->{tType})
    terra terra_expandAs(tArg1 : tType, tArg2 : tType) : tType
      var res = newWithTensorFun(tArg1)
      if(tArg1.size[0]==1) then
        res.size[0] = tArg2.size[0]
        res.stride[0] = 0
      end
      if(tArg1.size[1]==1) then
        res.size[1] = tArg2.size[1]
        res.stride[1] = 0
      end
      return res
    end
    list[torch.DoubleTensor.expandAs] = {
      terra_expandAs,
      {
        {name=Tensor,dim=2},
        {name=Tensor,dim=2},
        {name=Tensor,creturned=true}
      }
    }
    --TODO hacky hardcoding "torch.ByteTensor" as second tensor type
      C = terralib.includec("stdio.h")
    local copyFun = terralib.externfunction(cname("copyByte"),{tType,byteTensorType}->{})
    terra terra_util_typeAsInPlace(tArg1 : tType, tArg2 : byteTensorType, tArg3 : tType)
      copyFun(tArg1,tArg2)
    end
      
    list["util.typeAsInPlace"] = {
      terra_util_typeAsInPlace,
      {
        {name=Tensor, returned=true},
        {name="torch.ByteTensor"},
        {name=Tensor}
      }
    }
    local fillFun = terralib.externfunction(cname("fill"),{tType,double}->{})
    terra terra_util_fillInPlace(tArg1 : tType, tArg2 : tType, tArg3 : double)
      fillFun(tArg1,tArg3)
    end
      
    list["util.fillInPlace"] = {
      terra_util_fillInPlace,
      {
        {name=Tensor, returned=true},
        {name=Tensor},
        {name=real}
      }
    }
    
    -- Assume that it is contiguous
    --TODO copy data of tensor into result better (using unfold??)
    local isContig = terralib.externfunction(cname("isContiguous"),{tType}->{int32})
    local tensorResize2D = terralib.externfunction(cname("resize2d"),{tType,int32,int32}->{})
    local tensorSet2D = terralib.externfunction(cname("set2d"),{tType,int32,int32,double}->{})
    local tensorGet2D = terralib.externfunction(cname("get2d"),{tType,int32,int32}->{double})
    terra terra_torch_repeatTensor(result : tType, tensor : tType, size : longStorageType) : tType
      var new_size0 = tensor.size[0] * size.data[0]
      var new_size1 = tensor.size[1] * size.data[1]
      tensorResize2D(result,new_size0,new_size1) 
      for i0 = 0,size.data[0] do
        for i1 = 0,size.data[1] do
          for d0 = 0,tensor.size[0] do
            for d1 = 0,tensor.size[1] do
              var val = tensorGet2D(tensor,d0,d1)
              tensorSet2D(result,i0*tensor.size[0]+d0,i1*tensor.size[1]+d1,val)
            end
          end
        end
      end
    end
    list[torch.repeatTensor] = {
      terra_torch_repeatTensor,
      {
        {name=Tensor, returned=true},
        {name=Tensor},
        {name="torch.LongStorage"}
      }
    }


  name = nil
  cArgsList.list = list
end

cArgsList:create("torch.DoubleTensor")
cArgsList:checkCArgs()

return cArgsList
