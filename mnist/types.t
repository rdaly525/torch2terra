local th = terralib.includec("/mnt/raid/torch/install/include/TH/TH.h")

local types = {}

types.tensorTypes = {}
types.tensorTypes["torch.DoubleTensor"] = true
types.tensorTypes["torch.ByteTensor"] = true

types.torchTypes = {}
types.torchTypes["torch.DoubleTensor"] = &th.THDoubleTensor
types.torchTypes["torch.ByteTensor"] = &th.THByteTensor
types.torchTypes["double"] = double
types.torchTypes["index"] = int32

types.DoubleTensorStr="torch.DoubleTensor"
types.DoubleTensorType = &th.THDoubleTensor

types.ByteTensorStr="torch.ByteTensor"
types.ByteTensorType = &th.THByteTensor

types.realStr = "double"
types.realType = double


--local cArg = {}
--
--function cArg.new()
--  self = {}
--  setmetatable(self, {__index=tWrap})
--
--
--function cArg:


return types

