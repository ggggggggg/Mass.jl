using Mass
ljhgroup1=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")
ljhgroup1.ljhfiles[1].nrec = div(ljhgroup1.ljhfiles[1].nrec,2)
ljhgroup2 = Mass.MicrocalFiles.LJHGroup(ljhgroup1.ljhfiles[1])

h5 = h5open(hdf5_name_from_ljh(ljhgroup2),"w")
close(h5)
h5 = h5open(hdf5_name_from_ljh(ljhgroup2),"r+")

function ptm_correction(r, params, ptm, ph)
    ptm_offset, slope = params
    ph += (ptm.-ptm_offset).*ph
    return (ph,)
end
ptm_correction_step = Step(ptm_correction, "pretrigger_mean_correction", ["pretrig_mean","pulse_rms"], (), "pulse_rms_dc")

g = init_channel(h5, ljhgroup2)
a_update(g, "pretrigger_mean_correction", [2000.0, 0.1]) # imagine this is generated by another step that runs only once
h5step_add(g,summarize_step)
h5step_add(g, ptm_correction_step)
update!(g)
for j=1:3 update!(g,100000) end

