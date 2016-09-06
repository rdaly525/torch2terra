local th = terralib.includec("/mnt/raid/torch/install/include/TH/TH.h")

local types = {}

types.tensorTypes = {}
types.tensorTypes["torch.DoubleTensor"] = true
types.tensorTypes["torch.ByteTensor"] = true
types.tensorTypes["torch.LongTensor"] = true
types.tensorTypes["torch.LongStorage"] = true

types.torchTypes = {}
types.torchTypes["torch.DoubleTensor"] = &th.THDoubleTensor
types.torchTypes["torch.ByteTensor"] = &th.THByteTensor
types.torchTypes["torch.LongTensor"] = &th.THLongTensor
types.torchTypes["torch.LongStorage"] = &th.THLongStorage
types.torchTypes["torch.Number"] = double
types.torchTypes["double"] = double
types.torchTypes["index"] = int32

types.DoubleTensorStr="torch.DoubleTensor"
types.DoubleTensorType = &th.THDoubleTensor

types.ByteTensorStr="torch.ByteTensor"
types.ByteTensorType = &th.THByteTensor

types.realStr = "double"
types.realType = double

function types.getType(typeStr)
  if types.torchTypes[typeStr]==nil then
    assert(false,"ERROR: bad argType! "..typeStr)
  end
  return types.torchTypes[typeStr]
end

return types
