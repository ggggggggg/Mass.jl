module Mass
include("MicrocalFiles.jl")
include("H5Flow.jl")
include("Summarize.jl")
using .MicrocalFiles, .H5Flow, .Summarize, Logging
Logging.configure(level=DEBUG)
export microcal_open, hdf5_name_from_ljh, channel, record_nsamples, pretrig_nsamples, frametime, filenames, lengths,
	   summarize_step, summarize, init_channel, # from summarize
	   h5step_add, h5steps, h5stepnumbers, 
	   H5Flow, MicrocalFiles, Summarize,
	   g_require, # group stuff
       d_update, d_extend, d_require, #dataset stuff
       a_update, a_require, a_read, # attribute stuff
       hdf5_name_from_ljh_name, allnames,
       close, HDF5Group, HDF5File, name, attrs, names, jldopen,
       Step, update!, h5steps, h5step_add


println("yay imported Mass!")
end # module
