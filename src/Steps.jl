using Mass, H5Flow


ljh=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")
h5 = h5open(hdf5_name_from_ljh_name(ljh.name),"w")
close(h5)
@time o2=Mass.Summarize.summarize_flow(ljh)


h5 = h5open(hdf5_name_from_ljh_name(ljh.name),"r+")
g=g_require(h5,"chan1")
a_update(g, "pretrigger_mean_correction", [2000.0, 0.1])

function ptm_correction(params, ptm, ph)
    ptm_offset, slope = params
    ph += (ptm.-ptm_offset).*ph
    return (ph,)
end
s = Step("ptm_correction", "pretrigger_mean_correction", ["summary/pretrig_mean","summary/pulse_rms"], (), "pulse_rms_dc")
h5step_add(g,s, 10)
update!(g)

Mass.H5Flow.h5step_write(g,s)
s2 = Mass.H5Flow.h5step_read(g)
@assert(s2==s)