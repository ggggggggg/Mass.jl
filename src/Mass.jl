
include("H5Flow.jl")
include("MicrocalFiles.jl")
include("Summarize.jl")
module Mass

using Summarize, MicrocalFiles, H5Flow
export microcal_open, hdf5_name_from_ljh_name, channel, record_nsamples, pretrig_nsamples, timebase, filenames


println("yay imported Mass!")
end # module
