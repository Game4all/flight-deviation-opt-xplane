import xpc

connect = xpc.XPlaneConnect()

connect.sendTEXT("Testing les gars", -1, -1)
connect.sendVIEW(xpc.ViewType.Chase)
