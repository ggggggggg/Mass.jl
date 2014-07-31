using Mass, H5Flow, MicrocalFiles, Logging
Logging.configure(level=DEBUG)


ljh=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")
h5 = h5open(hdf5_name_from_ljh(ljh),"w")
close(h5)
h5 = h5open(hdf5_name_from_ljh(ljh),"r+")

function init_channel(h5::Union(HDF5File, HDF5Group), ljhgroup::LJHGroup)
	g = g_require(h5, "chan$(channel(ljhgroup))")
	a_require(g, "pulsefiles", filenames(ljhgroup))
	a_require(g, "npulses", length(ljhgroup))
	g
end
g = init_channel(h5, ljh)

function summarize(r::UnitRange,filenames)
	debug("summarize")
	ljhgroup = microcal_open(filenames)
	debug(ljhgroup)
	out = [mean(p) for (p,t) in ljhgroup[r]], [t for (p,t) in ljhgroup[r]]
	debug(typeof(out))
	out
end
s = Step(summarize, "pulsefiles", (), (), ["mean", "timestamp"])
h5step_add(g,s,10)
update!(g)




