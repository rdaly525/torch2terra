local cArgsList = {}


function cArgsList:checkCArgs()
  for op, v in pairs(cArgsList.list) do
    assert(#v % 2 == 0)
    for i,args in ipairs(v) do
      assert((i%2==1 and type(args)=="string") or (i%2==0 and type(args)=="table"))
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



function cArgsList:init(Tensor)
  
  list = {}
  TensorShort = Tensor:match("torch.(%S+)")
  local function cname(name)
    return string.format('TH%s_%s', TensorShort,name)
  end
  local real = reals[TensorShort]
  local accreal = accreals[TensorShort]
  
  
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
         {{name='ByteTensor',default=true, returned=true},
          {name=Tensor},
          {name=real}},
         cname(name .. 'ValueT'),
         {{name=Tensor, returned=true},
          {name=Tensor},
          {name=real}},
         cname(name .. 'Tensor'),
         {{name='ByteTensor',default=true, returned=true},
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
            {name="IndexTensor", default=true, returned=true, noreadadd=true},
            {name=Tensor},
            {name="index"}} }
  name = nil
  cArgsList.list = list
  
  --return cArgsList
end

local DoubleTensorStr="torch.DoubleTensor"

cArgsList:init(DoubleTensorStr)
cArgsList:checkCArgs()

return cArgsList
