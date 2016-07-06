local cwrapFun = {}



function cwrapFun:checkCwrap(cwrap)
  for op, v in pairs(cwrap) do
    assert(#v % 2 == 0)
    for i,args in ipairs(v) do
      assert((i%2==1 and type(args)=="string") or (i%2==0 and type(args)=="table"))
    end
    --Verify that creturned is always the last arg if it is there
    --for i=1,#v/2 do
    --  for ci,t in ipairs
    --end
  end
end


function cwrapFun:init(Tensor)
  local function cname(name)
    return string.format('TH%s_%s', Tensor,name)
  end
  local cwrap = {} 
  
  
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

  local real = reals[Tensor]
  local accreal = accreals[Tensor]
  
  
  cwrap[torch.add] = {
          cname("add"),
          {{name=Tensor, default=true, returned=true, method={default='nil'}},
           {name=Tensor, method={default=1}},
           {name=real}},
          cname("cadd"),
          {{name=Tensor, default=true, returned=true, method={default='nil'}},
           {name=Tensor, method={default=1}},
           {name=real, default=1},
           {name=Tensor}}}
  cwrap[torch.neg] = {
       cname("neg"),
       {{name=Tensor, default=true, returned=true, method={default='nil'}},
        {name=Tensor, method={default=1}}}}
  cwrap[torch.log] = {
       cname("log"),
       {{name=Tensor, default=true, returned=true, method={default='nil'}},
        {name=Tensor, method={default=1}}},
       "log",
       {{name=real},
        {name=real, creturned=true}}}
  cwrap[torch.cdiv] = {
        cname("cdiv"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=Tensor}}}
  cwrap[torch.cmul] = {
        cname("cmul"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}},
         {name=Tensor}}}
  cwrap[torch.pow] = {
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
  cwrap[torch.exp] = {
        cname("exp"),
        {{name=Tensor, default=true, returned=true, method={default='nil'}},
         {name=Tensor, method={default=1}}},
        "exp",
        {{name=real},
         {name=real, creturned=true}}}
  local name = "eq"
  cwrap[torch.eq] = {
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
  cwrap[torch.sum] = {
          cname("sumall"),
          {{name=Tensor},
           {name=accreal, creturned=true}},
          cname("sum"),
          {{name=Tensor, default=true, returned=true},
           {name=Tensor},
           {name="index"}} }
  
  name = "max"
  cwrap[torch.max] = {
           cname(name .. "all"),
           {{name=Tensor},
            {name=real, creturned=true}},
           cname(name),
           {{name=Tensor, default=true, returned=true},
            {name="IndexTensor", default=true, returned=true, noreadadd=true},
            {name=Tensor},
            {name="index"}} }
  name = nil
  return cwrap
end

return cwrapFun
