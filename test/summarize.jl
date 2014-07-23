using Mass

f=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")

@time o1=Mass.Summarize.compute_summary_iter(f)
@time o2=Mass.Summarize.compute_summary_iter(f)
@time o3=Mass.Summarize.compute_summary(f)
@time o4=Mass.Summarize.compute_summary(f)
println(o1.pretrig_mean==o2.pretrig_mean)
println(o1.pretrig_mean==o3.pretrig_mean)
println(o1.pretrig_mean==o4.pretrig_mean)