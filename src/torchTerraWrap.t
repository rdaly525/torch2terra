local luaT_toudata = terralib.externfunction("luaT_pushudata", {&opaque, &opaque, rawstring} -> int)
local lua_topointer = terralib.externfunction("lua_topointer", {&opaque, int} -> &opaque)
local lua_tolstring = terralib.externfunction("lua_tolstring", {&opaque, int,&opaque} -> rawstring)
local c = terralib.includec("stdio.h")
terra wrapTorchObjectT(L : &opaque)
    var cdata = [&&opaque](lua_topointer(L,1))
    luaT_toudata(L,@cdata,lua_tolstring(L,2,nil))
    return 1
end

--wrapTorchObject(cData,torchTypeStaring)
wrapTorchObject = terralib.bindtoluaapi(wrapTorchObjectT:getpointer())

--unwrapTorchObject(torchObject,torchObjectType)
function unwrapTorchObject(obj,t)
  if(t) then
    return terralib.cast(&t,obj)[0]
  else
    return terralib.cast(&&opaque,obj)[0]
  end
end
