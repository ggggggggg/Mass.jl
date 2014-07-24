using Mass, Logging
Logging.configure(level=DEBUG)

f=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")

@time o1=Mass.Summarize.compute_summary(f)

f.nrec = div(f.nrec,2)

@time o2=Mass.Summarize.summarize_flow(f, true)
@show Mass.MicrocalFiles.update_num_records(f)
@time o2=Mass.Summarize.summarize_flow(f)
@time o2=Mass.Summarize.summarize_flow(f)
