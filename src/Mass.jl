
include("H5Flow.jl")
include("MicrocalFiles.jl")
include("Summarize.jl")
module Mass
using MicrocalFiles, H5Flow, Logging
Logging.configure(level=DEBUG)
export microcal_open, hdf5_name_from_ljh, channel, record_nsamples, pretrig_nsamples, frametime, filenames, lengths


println("yay imported Mass!")
end # module
