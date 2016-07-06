--Strict.strict = false
local th = terralib.includec("/mnt/raid/torch/install/include/TH/TH.h")

funMap = {}

--Comment

--funMap[torch.add] = function(a,b,c)
--  cast_a = terralib.cast(&&th.THDoubleTensor,a)
--  cast_b = terralib.cast(&&th.THDoubleTensor,b)
--  cast_c = terralib.cast(&&th.THDoubleTensor,c)
--  th.THDoubleTensor_cadd(cast_a[0],cast_b[0],0,cast_c[0])
--  return a
--end

luaT_toudata(L,1,"torch.DoubleTensor")

--/mnt/raid/torch/pkg/torch/TensorMath.lua contains generation code


funMap[assert(torch.neg)] = function(a,b)
  cast_a = terralib.cast(&&th.THDoubleTensor,a)
  cast_b = terralib.cast(&&th.THDoubleTensor,b)
  th.THDoubleTensor_neg(cast_a[0],cast_b[0])
  return a
end

--funMap[assert(torch.sub)] = function(a,b,c)
--  cast_a = terralib.cast(&&th.THDoubleTensor,a)
--  cast_b = terralib.cast(&&th.THDoubleTensor,b)
--  th.THDoubleTensor_sub(cast_a[0],cast_b[0],c)
--end

--funMap[torch.cmul] = function(a,b,c)
--  cast_a = terralib.cast(&&th.THDoubleTensor,a)
--  cast_b = terralib.cast(&&th.THDoubleTensor,b)
--  cast_c = terralib.cast(&&th.THDoubleTensor,c)
--  th.THDoubleTensor_cmul(cast_a[0],cast_b[0],cast_c[0])
--end
--
--funMap[torch.cdiv] = function(a,b,c)
--  cast_a = terralib.cast(&&th.THDoubleTensor,a)
--  cast_b = terralib.cast(&&th.THDoubleTensor,b)
--  cast_c = terralib.cast(&&th.THDoubleTensor,c)
--  th.THDoubleTensor_cdiv(cast_a[0],cast_b[0],cast_c[0])
--end

