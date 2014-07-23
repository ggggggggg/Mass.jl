using Mass

f=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")

@time [p for p in f[1:end]]
f=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")
@time Mass.MicrocalFiles.LJH.fileData(f)
