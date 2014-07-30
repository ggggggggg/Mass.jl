module LJHTest
using Mass, Base.Test

f=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")

@time [p for p in f[1:end]]
@time Mass.MicrocalFiles.LJH.fileData(f)

ljhgroup = Mass.MicrocalFiles.LJH.LJHGroup(["/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan$d.ljh" for d in 1:2:5])
gs = ljhgroup[1:100]
@time [p for p in gs]

a=[p for p in f[1:100]]
a2 = [p for p in ljhgroup[1:100]]
@test a == a2
a3 = [p for p in ljhgroup[1:end]]
end